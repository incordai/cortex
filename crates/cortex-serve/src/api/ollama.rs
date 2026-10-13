//! Ollama-compatible `/api/generate`, `/api/chat`, `/api/tags` and
//! `/api/version`.
//!
//! Ollama streams by default, as newline-delimited JSON; `"stream": false`
//! returns one object. Sampling settings arrive in `options`
//! (`num_predict`, `repeat_penalty`, ...), and `format` is `"json"` or a
//! JSON schema, enforced token by token. A streamed chat sends each tool
//! call as soon as it is complete, in Ollama's whole-call form.

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::{
    ApiError, AppState, JsonMode, Sampling, Spec, collect, failure, ndjson_response, prepare, render, resolve, submit,
    tool_ids, tool_names, tool_parser, unix_now,
};
use crate::engine::{Event, FinishReason, Usage};
use crate::model::{ChatMessage, Role, ToolCall, ToolSpec};
use crate::tools::{ToolEvent, parse_tool_calls};

#[derive(Deserialize, Default)]
struct Options {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    num_predict: Option<i64>,
    stop: Option<Vec<String>>,
    seed: Option<u64>,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
}

impl Options {
    fn sampling(&self) -> Sampling {
        Sampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
        }
    }

    /// `num_predict` (negative means "no limit", which the server's default
    /// bounds).
    fn max_tokens(&self) -> Option<usize> {
        self.num_predict.and_then(|n| usize::try_from(n).ok())
    }
}

#[derive(Deserialize)]
pub(super) struct GenerateRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: String,
    #[serde(default)]
    raw: bool,
    stream: Option<bool>,
    format: Option<Value>,
    #[serde(default)]
    options: Options,
}

#[derive(Deserialize)]
struct ChatToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Deserialize)]
struct ChatToolCall {
    function: ChatToolCallFunction,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<ChatToolCall>,
}

#[derive(Deserialize)]
struct Tool {
    function: ToolSpec,
}

#[derive(Deserialize)]
pub(super) struct ChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<Message>,
    stream: Option<bool>,
    format: Option<Value>,
    #[serde(default)]
    options: Options,
    #[serde(default)]
    tools: Vec<Tool>,
}

fn json_mode(format: Option<&Value>) -> Result<JsonMode, ApiError> {
    match format {
        None | Some(Value::Null) => Ok(JsonMode::Off),
        Some(Value::String(s)) if s.is_empty() => Ok(JsonMode::Off),
        Some(Value::String(s)) if s == "json" => Ok(JsonMode::any_value()),
        Some(Value::String(other)) => Err(ApiError::bad_request(format!("format '{other}' is not supported"))),
        Some(schema) => JsonMode::schema(schema),
    }
}

/// `seconds` since the epoch as RFC 3339 UTC.
fn rfc3339(seconds: u64) -> String {
    // Civil-from-days (H. Hinnant), valid for every date after 1970.
    let days = (seconds / 86_400) as i64;
    let secs = seconds % 86_400;
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs / 60) % 60,
        secs % 60
    )
}

fn done_reason(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::StopString(_) => "stop",
        FinishReason::Length => "length",
    }
}

fn final_fields(v: &mut Value, reason: &FinishReason, usage: &Usage, started: Instant) {
    let total = started.elapsed().as_nanos() as u64;
    v["done"] = json!(true);
    v["done_reason"] = json!(done_reason(reason));
    v["total_duration"] = json!(total);
    v["prompt_eval_count"] = json!(usage.prompt_tokens);
    v["eval_count"] = json!(usage.completion_tokens);
}

fn tool_calls_json(calls: &[ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| json!({"function": {"name": c.name, "arguments": c.arguments}}))
            .collect(),
    )
}

/// `GET /api/tags`.
pub(super) async fn tags(State(state): State<Arc<AppState>>) -> Json<Value> {
    let models: Vec<Value> = state
        .registry
        .list()
        .into_iter()
        .map(|(name, created)| {
            json!({
                "name": name, "model": name, "modified_at": rfc3339(created),
                "size": 0, "digest": "",
                "details": {"format": "cortex", "family": "", "parameter_size": "", "quantization_level": ""},
            })
        })
        .collect();
    Json(json!({"models": models}))
}

/// `GET /api/version`.
pub(super) async fn version() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION")}))
}

/// `POST /api/generate`.
pub(super) async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let (name, model) = resolve(&state, &req.model).await?;
    let (prompt, special) = if req.raw {
        (req.prompt, false)
    } else {
        let mut chat = Vec::new();
        if !req.system.is_empty() {
            chat.push(ChatMessage::text(Role::System, req.system));
        }
        chat.push(ChatMessage::text(Role::User, req.prompt));
        (render(&model, &chat, &[])?, true)
    };
    let prepared = prepare(
        &model,
        &state.limits,
        Spec {
            prompt: &prompt,
            special,
            sampling: req.options.sampling(),
            default_temperature: 0.8,
            max_tokens: req.options.max_tokens(),
            stop: req.options.stop.clone().unwrap_or_default(),
            json: json_mode(req.format.as_ref())?,
            tools: Vec::new(),
        },
    )?;
    let base = move |name: &str| json!({"model": name, "created_at": rfc3339(unix_now())});

    if req.stream == Some(false) {
        let out = collect(&model, prepared.request).await?;
        let mut v = base(&name);
        v["response"] = json!(out.text);
        final_fields(&mut v, &out.reason, &out.usage, started);
        return Ok(Json(v).into_response());
    }

    let mut events = submit(&model, prepared.request)?;
    let (tx, rx) = mpsc::channel::<Value>(64);
    tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            let mut v = base(&name);
            match event {
                Event::Text(t) => {
                    v["response"] = json!(t);
                    v["done"] = json!(false);
                    if tx.send(v).await.is_err() {
                        return;
                    }
                }
                Event::Done { reason, usage } => {
                    v["response"] = json!("");
                    final_fields(&mut v, &reason, &usage, started);
                    let _ = tx.send(v).await;
                    return;
                }
                failed @ (Event::Error(_) | Event::Failed(_)) => {
                    let _ = tx.send(json!({"error": failure(failed).1})).await;
                    return;
                }
            }
        }
    });
    Ok(ndjson_response(rx))
}

/// `POST /api/chat`.
pub(super) async fn chat(State(state): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Result<Response, ApiError> {
    let started = Instant::now();
    let (name, model) = resolve(&state, &req.model).await?;
    let mut ids = tool_ids();
    let chat: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                other => return Err(ApiError::bad_request(format!("unknown message role '{other}'"))),
            };
            Ok(ChatMessage {
                role,
                content: m.content,
                tool_calls: m
                    .tool_calls
                    .into_iter()
                    .map(|c| ToolCall {
                        id: ids(),
                        name: c.function.name,
                        arguments: c.function.arguments,
                    })
                    .collect(),
                tool_call_id: None,
            })
        })
        .collect::<Result<_, _>>()?;
    let tools: Vec<ToolSpec> = req.tools.into_iter().map(|t| t.function).collect();
    let prompt = render(&model, &chat, &tools)?;
    let prepared = prepare(
        &model,
        &state.limits,
        Spec {
            prompt: &prompt,
            special: true,
            sampling: req.options.sampling(),
            default_temperature: 0.8,
            max_tokens: req.options.max_tokens(),
            stop: req.options.stop.clone().unwrap_or_default(),
            json: json_mode(req.format.as_ref())?,
            tools: tool_names(&tools),
        },
    )?;
    let base = move |name: &str| json!({"model": name, "created_at": rfc3339(unix_now())});

    if req.stream == Some(false) {
        let names = prepared.tools;
        let out = collect(&model, prepared.request).await?;
        let (content, calls) = if names.is_empty() {
            (out.text, Vec::new())
        } else {
            parse_tool_calls(&out.text, &names, tool_ids())
        };
        let mut v = base(&name);
        v["message"] = json!({"role": "assistant", "content": content});
        if !calls.is_empty() {
            v["message"]["tool_calls"] = tool_calls_json(&calls);
        }
        final_fields(&mut v, &out.reason, &out.usage, started);
        return Ok(Json(v).into_response());
    }

    let mut parser = tool_parser(&prepared.tools);
    let mut events = submit(&model, prepared.request)?;
    let (tx, rx) = mpsc::channel::<Value>(64);
    tokio::spawn(async move {
        // One NDJSON line per text piece or completed call.
        let part = |event: ToolEvent| -> Option<Value> {
            let mut v = base(&name);
            v["done"] = json!(false);
            match event {
                ToolEvent::Text(t) if !t.is_empty() => v["message"] = json!({"role": "assistant", "content": t}),
                ToolEvent::CallEnd { call: Some(call), .. } => {
                    v["message"] = json!({"role": "assistant", "content": "", "tool_calls": tool_calls_json(&[call])});
                }
                // A call that never became valid is shown as the text it was.
                ToolEvent::CallEnd { call: None, raw, .. } => v["message"] = json!({"role": "assistant", "content": raw}),
                _ => return None,
            }
            Some(v)
        };
        while let Some(event) = events.recv().await {
            let mut v = base(&name);
            match event {
                Event::Text(t) => {
                    let pieces = match parser.as_mut() {
                        Some(p) => p.push(&t),
                        None => vec![ToolEvent::Text(t)],
                    };
                    for line in pieces.into_iter().filter_map(part) {
                        if tx.send(line).await.is_err() {
                            return;
                        }
                    }
                }
                Event::Done { reason, usage } => {
                    if let Some(p) = parser.as_mut() {
                        for line in p.finish().into_iter().filter_map(part) {
                            if tx.send(line).await.is_err() {
                                return;
                            }
                        }
                    }
                    v["message"] = json!({"role": "assistant", "content": ""});
                    final_fields(&mut v, &reason, &usage, started);
                    let _ = tx.send(v).await;
                    return;
                }
                failed @ (Event::Error(_) | Event::Failed(_)) => {
                    let _ = tx.send(json!({"error": failure(failed).1})).await;
                    return;
                }
            }
        }
    });
    Ok(ndjson_response(rx))
}

#[cfg(test)]
mod tests {
    use super::rfc3339;

    #[test]
    fn timestamps_render_as_utc() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        // 1767225600 is 2026-01-01; 24631197 s later is day 285 (13 Oct)
        // plus 7197 s.
        assert_eq!(rfc3339(1_791_856_797), "2026-10-13T01:59:57Z");
    }
}
