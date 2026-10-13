//! The HTTP surface: OpenAI (`/v1/chat/completions`, `/v1/completions`,
//! `/v1/models`), Anthropic (`/v1/messages`) and Ollama (`/api/generate`,
//! `/api/chat`, `/api/tags`, `/api/version`) protocols over one set of
//! engines.
//!
//! Every handler does the same three things: resolve the model in the
//! [`Registry`], turn the protocol's request into a [`GenerationRequest`]
//! ([`Prepared`]), and either collect the engine's events into one response
//! or translate them into the protocol's stream.
//!
//! Callers: `server.rs` mounts [`router`]. Calls: `registry.rs`,
//! `engine.rs`, `constrain.rs`, `tools.rs`.

mod anthropic;
mod ollama;
mod openai;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use cortex_llm::SamplerConfig;
use serde_json::json;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::constrain::{Constraint, JsonConstraint};
use crate::engine::{Event, FinishReason, GenerationRequest, Usage};
use crate::model::{ChatMessage, ToolSpec};
use crate::registry::{Registry, RegistryError, ServedModel};
use crate::schema::Schema;
use crate::tools::ToolParser;

/// Server-wide request limits.
#[derive(Clone, Debug)]
pub struct Limits {
    /// `max_tokens` when a request does not give one.
    pub default_max_tokens: usize,
    /// The largest `max_tokens` a request may ask for.
    pub max_tokens_limit: usize,
    /// The longest prompt accepted, in tokens.
    pub max_prompt_tokens: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            default_max_tokens: 2048,
            max_tokens_limit: 32768,
            max_prompt_tokens: 131072,
        }
    }
}

/// State shared by every handler.
pub struct AppState {
    /// The served models.
    pub registry: Registry,
    /// Request limits.
    pub limits: Limits,
}

/// Mounts every endpoint.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/messages", post(anthropic::messages))
        .route("/api/generate", post(ollama::generate))
        .route("/api/chat", post(ollama::chat))
        .route("/api/tags", get(ollama::tags))
        .route("/api/version", get(ollama::version))
        .with_state(state)
}

/// An error in the shape every protocol understands well enough:
/// `{"error": {"message", "type"}}` with a matching status.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: message.into(),
        }
    }
}

impl From<RegistryError> for ApiError {
    fn from(e: RegistryError) -> Self {
        match e {
            RegistryError::NotFound(name) => Self {
                status: StatusCode::NOT_FOUND,
                kind: "not_found_error",
                message: format!("model '{name}' is not served here"),
            },
            RegistryError::LoadFailed(why) => Self::internal(format!("the model failed to load: {why}")),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({"error": {"message": self.message, "type": self.kind}}));
        (self.status, body).into_response()
    }
}

/// Resolves `name` in the registry without blocking the async runtime.
async fn resolve(state: &Arc<AppState>, name: &str) -> Result<(String, Arc<ServedModel>), ApiError> {
    let state = state.clone();
    let name = name.to_string();
    tokio::task::spawn_blocking(move || state.registry.get(&name))
        .await
        .map_err(|e| ApiError::internal(format!("model lookup failed: {e}")))?
        .map_err(ApiError::from)
}

/// Sampling fields common to every protocol, all optional.
#[derive(Clone, Debug, Default)]
struct Sampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
}

impl Sampling {
    /// The sampler settings, validated; `default_temperature` is the
    /// protocol's documented default.
    fn config(&self, default_temperature: f32) -> Result<SamplerConfig, ApiError> {
        let temperature = self.temperature.unwrap_or(default_temperature);
        if !(0.0..=5.0).contains(&temperature) {
            return Err(ApiError::bad_request("temperature must be between 0 and 5"));
        }
        let top_p = self.top_p.unwrap_or(1.0);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(ApiError::bad_request("top_p must be in (0, 1]"));
        }
        let min_p = self.min_p.unwrap_or(0.0);
        if !(0.0..=1.0).contains(&min_p) {
            return Err(ApiError::bad_request("min_p must be in [0, 1]"));
        }
        let repetition_penalty = self.repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            return Err(ApiError::bad_request("repetition_penalty must be positive"));
        }
        for (name, v) in [("presence_penalty", self.presence_penalty), ("frequency_penalty", self.frequency_penalty)] {
            if let Some(v) = v
                && !(-2.0..=2.0).contains(&v)
            {
                return Err(ApiError::bad_request(format!("{name} must be between -2 and 2")));
            }
        }
        Ok(SamplerConfig {
            temperature,
            top_k: self.top_k.unwrap_or(0),
            top_p,
            min_p,
            repetition_penalty,
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            seed: self.seed.unwrap_or_else(random_seed),
            ..SamplerConfig::default()
        })
    }
}

fn random_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    nanos ^ COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Whether the output must be JSON, and of which schema.
#[derive(Clone, Debug)]
enum JsonMode {
    Off,
    Schema(Arc<Schema>),
}

impl JsonMode {
    /// Any JSON value.
    fn any_value() -> Self {
        Self::Schema(Arc::new(Schema::any_value()))
    }

    /// Any JSON object.
    fn any_object() -> Self {
        Self::Schema(Arc::new(Schema::any_object()))
    }

    /// JSON matching the JSON Schema `schema`.
    fn schema(schema: &serde_json::Value) -> Result<Self, ApiError> {
        Schema::compile(schema)
            .map(|s| Self::Schema(Arc::new(s)))
            .map_err(|e| ApiError::bad_request(format!("the JSON schema cannot be enforced: {e}")))
    }
}

/// Which tool, if any, the model must call.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolChoice {
    /// The model decides.
    Auto,
    /// Tools are not offered.
    None,
    /// Some tool must be called.
    Required,
    /// This tool must be called.
    Named(String),
}

/// The tools a request offers after `choice`, and the output constraint a
/// forced choice imposes (`json` must then be off).
fn apply_tool_choice(tools: Vec<ToolSpec>, choice: &ToolChoice, json: JsonMode) -> Result<(Vec<ToolSpec>, JsonMode), ApiError> {
    let forced: Vec<&ToolSpec> = match choice {
        ToolChoice::None => return Ok((Vec::new(), json)),
        ToolChoice::Auto => return Ok((tools, json)),
        ToolChoice::Required => tools.iter().collect(),
        ToolChoice::Named(name) => {
            let tool = tools
                .iter()
                .find(|t| &t.name == name)
                .ok_or_else(|| ApiError::bad_request(format!("tool_choice names '{name}', which is not among the tools")))?;
            vec![tool]
        }
    };
    if forced.is_empty() {
        return Err(ApiError::bad_request("tool_choice requires a tool call but no tools are given"));
    }
    if matches!(json, JsonMode::Schema(_)) {
        return Err(ApiError::bad_request("a forced tool call cannot be combined with a JSON response format"));
    }
    let specs: Vec<(&str, &serde_json::Value)> = forced.iter().map(|t| (t.name.as_str(), &t.parameters)).collect();
    let schema = Schema::tool_calls(&specs).map_err(|e| ApiError::bad_request(format!("cannot force the tool call: {e}")))?;
    Ok((tools, JsonMode::Schema(Arc::new(schema))))
}

/// The names of `tools`, for [`ToolParser`].
fn tool_names(tools: &[ToolSpec]) -> Vec<String> {
    tools.iter().map(|t| t.name.clone()).collect()
}

/// A streaming tool-call parser when tools are offered.
fn tool_parser(names: &[String]) -> Option<ToolParser> {
    (!names.is_empty()).then(|| ToolParser::new(names.to_vec(), tool_ids()))
}

/// What every handler needs to run a generation.
struct Prepared {
    request: GenerationRequest,
    /// Names of the tools offered (empty: no tool calls are parsed).
    tools: Vec<String>,
}

/// A protocol request reduced to what the engine needs.
struct Spec<'a> {
    /// Rendered chat prompt, or raw completion text.
    prompt: &'a str,
    /// Whether special-token markup in `prompt` is honoured (templates).
    special: bool,
    sampling: Sampling,
    /// The protocol's documented default temperature.
    default_temperature: f32,
    max_tokens: Option<usize>,
    stop: Vec<String>,
    json: JsonMode,
    tools: Vec<String>,
}

/// Builds the engine request for a rendered or raw prompt, enforcing the
/// server's limits.
fn prepare(model: &ServedModel, limits: &Limits, spec: Spec<'_>) -> Result<Prepared, ApiError> {
    let Spec {
        prompt,
        special,
        sampling,
        default_temperature,
        max_tokens,
        stop,
        json,
        tools,
    } = spec;
    let tokens = model
        .tokenizer
        .encode(prompt, special)
        .map_err(|why| ApiError::bad_request(format!("the prompt could not be tokenized: {why}")))?;
    if tokens.is_empty() {
        return Err(ApiError::bad_request("the prompt is empty"));
    }
    if tokens.len() > limits.max_prompt_tokens {
        return Err(ApiError::bad_request(format!(
            "the prompt has {} tokens; the limit is {}",
            tokens.len(),
            limits.max_prompt_tokens
        )));
    }
    let max_tokens = max_tokens.unwrap_or(limits.default_max_tokens);
    if max_tokens > limits.max_tokens_limit {
        return Err(ApiError::bad_request(format!(
            "max_tokens may be at most {}",
            limits.max_tokens_limit
        )));
    }
    let constraint: Option<Box<dyn Constraint>> = match json {
        JsonMode::Off => None,
        JsonMode::Schema(schema) => Some(Box::new(JsonConstraint::with_schema(model.tokens.clone(), schema))),
    };
    Ok(Prepared {
        request: GenerationRequest {
            prompt: tokens,
            sampling: sampling.config(default_temperature)?,
            max_tokens,
            stop_tokens: model.tokenizer.stop_tokens(),
            stop_strings: stop,
            constraint,
        },
        tools,
    })
}

/// Renders a chat with the model's template; a conversation the template
/// refuses is the client's error.
fn render(model: &ServedModel, messages: &[ChatMessage], tools: &[ToolSpec]) -> Result<String, ApiError> {
    model
        .template
        .render(messages, tools)
        .map_err(|why| ApiError::bad_request(format!("the model's chat template refused the conversation: {why}")))
}

/// The whole output of a generation.
struct Collected {
    text: String,
    reason: FinishReason,
    usage: Usage,
}

/// Runs `request` to completion.
async fn collect(model: &ServedModel, request: GenerationRequest) -> Result<Collected, ApiError> {
    let mut rx = submit(model, request)?;
    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            Event::Text(t) => text.push_str(&t),
            Event::Done { reason, usage } => return Ok(Collected { text, reason, usage }),
            Event::Error(e) => return Err(ApiError::bad_request(e)),
            Event::Failed(e) => return Err(ApiError::internal(e)),
        }
    }
    Err(ApiError::internal("the engine stopped before finishing the request"))
}

/// The error type and message of an [`Event::Error`] (the request's fault)
/// or [`Event::Failed`] (the server's), for streamed error payloads.
fn failure(event: Event) -> (&'static str, String) {
    match event {
        Event::Error(e) => ("invalid_request_error", e),
        Event::Failed(e) => ("server_error", e),
        other => ("server_error", format!("unexpected engine event {other:?}")),
    }
}

fn submit(model: &ServedModel, request: GenerationRequest) -> Result<UnboundedReceiver<Event>, ApiError> {
    model
        .engine
        .submit(request)
        .map_err(|_| ApiError::internal("the model's engine has stopped"))
}

/// A server-sent-events response fed by a channel. When the client goes
/// away the stream is dropped, the producer's sends fail, it drops its
/// engine receiver, and the engine cancels the generation.
fn sse_response(rx: tokio::sync::mpsc::Receiver<SseEvent>) -> Response {
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|e| (Ok::<_, std::convert::Infallible>(e), rx))
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

/// A newline-delimited JSON response fed by a channel (Ollama streaming);
/// cancellation works as for [`sse_response`].
fn ndjson_response(rx: tokio::sync::mpsc::Receiver<serde_json::Value>) -> Response {
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| {
            let mut line = v.to_string();
            line.push('\n');
            (Ok::<_, std::convert::Infallible>(line), rx)
        })
    });
    (
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(stream),
    )
        .into_response()
}

/// Fresh tool-call identifiers.
fn tool_ids() -> impl FnMut() -> String + Send + 'static {
    || response_id("call_")
}

/// Seconds since the Unix epoch.
fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A unique response id with `prefix`.
fn response_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!("{prefix}{:x}{:06x}", unix_now(), COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// A `stop` field given as one string or a list.
fn stop_list(stop: Option<serde_json::Value>) -> Result<Vec<String>, ApiError> {
    match stop {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::String(s)) => Ok(vec![s]),
        Some(serde_json::Value::Array(items)) => items
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => Ok(s),
                _ => Err(ApiError::bad_request("stop must be a string or a list of strings")),
            })
            .collect(),
        Some(_) => Err(ApiError::bad_request("stop must be a string or a list of strings")),
    }
}

/// Text of a message content given as a string or a list of typed parts.
fn content_text(content: &serde_json::Value) -> Result<String, ApiError> {
    match content {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => out.push_str(part.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                    Some(other) => {
                        return Err(ApiError::bad_request(format!("content parts of type '{other}' are not supported")));
                    }
                    None => return Err(ApiError::bad_request("a content part has no type")),
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request("content must be a string or a list of parts")),
    }
}
