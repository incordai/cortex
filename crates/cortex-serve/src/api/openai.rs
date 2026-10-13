//! OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`,
//! `/v1/models`.
//!
//! Streaming follows the OpenAI chunk format (`chat.completion.chunk`,
//! terminated by `data: [DONE]`), including `stream_options.include_usage`.
//! Tool calls stream as `tool_calls` deltas: the call's id and name as soon
//! as the name is written, then its arguments piece by piece.
//! `response_format` `json_schema` is enforced token by token, and
//! `tool_choice` `required` or a named function forces a valid call.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::Event as SseEvent;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::{
    ApiError, AppState, JsonMode, Sampling, Spec, ToolChoice, apply_tool_choice, collect, content_text, failure, prepare,
    render, resolve, response_id, sse_response, stop_list, submit, tool_ids, tool_names, tool_parser, unix_now,
};
use crate::engine::{Event, FinishReason, Usage};
use crate::model::{ChatMessage, Role, ToolCall, ToolSpec};
use crate::tools::{ToolEvent, parse_tool_calls};

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct MessageToolCall {
    #[serde(default)]
    id: String,
    function: FunctionCall,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Vec<MessageToolCall>,
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
struct Tool {
    #[serde(rename = "type", default)]
    kind: String,
    function: ToolSpec,
}

#[derive(Deserialize)]
pub(super) struct ChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<Value>,
    n: Option<usize>,
    response_format: Option<Value>,
    #[serde(default)]
    tools: Vec<Tool>,
    tool_choice: Option<Value>,
}

#[derive(Deserialize)]
pub(super) struct CompletionRequest {
    #[serde(default)]
    model: String,
    prompt: Value,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<Value>,
    n: Option<usize>,
}

fn role(name: &str) -> Result<Role, ApiError> {
    match name {
        "system" | "developer" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(ApiError::bad_request(format!("unknown message role '{other}'"))),
    }
}

fn messages(raw: Vec<Message>) -> Result<Vec<ChatMessage>, ApiError> {
    raw.into_iter()
        .map(|m| {
            Ok(ChatMessage {
                role: role(&m.role)?,
                content: content_text(&m.content)?,
                tool_calls: m
                    .tool_calls
                    .into_iter()
                    .map(|c| ToolCall {
                        id: c.id,
                        name: c.function.name,
                        // Keep unparseable arguments as the string they were.
                        arguments: serde_json::from_str(&c.function.arguments)
                            .unwrap_or(Value::String(c.function.arguments)),
                    })
                    .collect(),
                tool_call_id: m.tool_call_id,
            })
        })
        .collect()
}

fn json_mode(format: Option<&Value>) -> Result<JsonMode, ApiError> {
    let Some(format) = format else {
        return Ok(JsonMode::Off);
    };
    match format.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(JsonMode::Off),
        Some("json_object") => Ok(JsonMode::any_object()),
        Some("json_schema") => {
            let schema = format
                .pointer("/json_schema/schema")
                .ok_or_else(|| ApiError::bad_request("response_format json_schema needs json_schema.schema"))?;
            JsonMode::schema(schema)
        }
        Some(other) => Err(ApiError::bad_request(format!("response_format '{other}' is not supported"))),
    }
}

fn tool_choice(choice: Option<&Value>) -> Result<ToolChoice, ApiError> {
    match choice {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Ok(ToolChoice::Required),
            other => Err(ApiError::bad_request(format!("tool_choice '{other}' is not supported"))),
        },
        Some(v) => v
            .pointer("/function/name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string()))
            .ok_or_else(|| ApiError::bad_request("tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}")),
    }
}

/// The chunk delta for one tool-parser event (`None`: nothing to send).
fn tool_delta(event: ToolEvent) -> Option<Value> {
    match event {
        ToolEvent::Text(t) if t.is_empty() => None,
        ToolEvent::Text(t) => Some(json!({"content": t})),
        ToolEvent::CallStart { index, id, name } => Some(json!({"tool_calls": [{
            "index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""},
        }]})),
        ToolEvent::CallArgs { index, delta } => {
            Some(json!({"tool_calls": [{"index": index, "function": {"arguments": delta}}]}))
        }
        ToolEvent::CallEnd { .. } => None,
    }
}

fn finish_reason(reason: &FinishReason, tool_calls: bool) -> &'static str {
    if tool_calls {
        return "tool_calls";
    }
    match reason {
        FinishReason::Stop | FinishReason::StopString(_) => "stop",
        FinishReason::Length => "length",
    }
}

fn usage_json(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
    })
}

fn tool_calls_json(calls: &[ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments.to_string()},
                })
            })
            .collect(),
    )
}

/// `GET /v1/models`.
pub(super) async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = state
        .registry
        .list()
        .into_iter()
        .map(|(id, created)| json!({"id": id, "object": "model", "created": created, "owned_by": "cortex"}))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

/// `POST /v1/chat/completions`.
pub(super) async fn chat(State(state): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Result<Response, ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("only n = 1 is supported"));
    }
    let choice = tool_choice(req.tool_choice.as_ref())?;
    let offered: Vec<ToolSpec> = req
        .tools
        .into_iter()
        .filter(|t| t.kind.is_empty() || t.kind == "function")
        .map(|t| t.function)
        .collect();
    let (tools, json) = apply_tool_choice(offered, &choice, json_mode(req.response_format.as_ref())?)?;
    let (name, model) = resolve(&state, &req.model).await?;
    let chat = messages(req.messages)?;
    let prompt = render(&model, &chat, &tools)?;
    let prepared = prepare(
        &model,
        &state.limits,
        Spec {
            prompt: &prompt,
            special: true,
            sampling: Sampling {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                min_p: req.min_p,
                repetition_penalty: req.repetition_penalty,
                presence_penalty: req.presence_penalty,
                frequency_penalty: req.frequency_penalty,
                seed: req.seed,
            },
            default_temperature: 1.0,
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            stop: stop_list(req.stop)?,
            json,
            tools: tool_names(&tools),
        },
    )?;
    let id = response_id("chatcmpl-");
    let created = unix_now();

    if !req.stream {
        let names = prepared.tools;
        let out = collect(&model, prepared.request).await?;
        let (content, calls) = if names.is_empty() {
            (out.text, Vec::new())
        } else {
            parse_tool_calls(&out.text, &names, tool_ids())
        };
        let mut message = json!({"role": "assistant", "content": content});
        if !calls.is_empty() {
            message["tool_calls"] = tool_calls_json(&calls);
        }
        return Ok(Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": name,
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": null,
                "finish_reason": finish_reason(&out.reason, !calls.is_empty()),
            }],
            "usage": usage_json(&out.usage),
        }))
        .into_response());
    }

    let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
    let mut parser = tool_parser(&prepared.tools);
    let mut events = submit(&model, prepared.request)?;
    let (tx, rx) = mpsc::channel::<SseEvent>(64);
    tokio::spawn(async move {
        let chunk = |delta: Value, finish: Option<&str>| {
            let v = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": name,
                "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish}],
            });
            SseEvent::default().data(v.to_string())
        };
        if tx.send(chunk(json!({"role": "assistant", "content": ""}), None)).await.is_err() {
            return;
        }
        while let Some(event) = events.recv().await {
            match event {
                Event::Text(t) => {
                    let pieces = match parser.as_mut() {
                        Some(p) => p.push(&t),
                        None => vec![ToolEvent::Text(t)],
                    };
                    for delta in pieces.into_iter().filter_map(tool_delta) {
                        if tx.send(chunk(delta, None)).await.is_err() {
                            return;
                        }
                    }
                }
                Event::Done { reason, usage } => {
                    let mut called = false;
                    if let Some(p) = parser.as_mut() {
                        for delta in p.finish().into_iter().filter_map(tool_delta) {
                            if tx.send(chunk(delta, None)).await.is_err() {
                                return;
                            }
                        }
                        called = p.has_calls();
                    }
                    let _ = tx.send(chunk(json!({}), Some(finish_reason(&reason, called)))).await;
                    if include_usage {
                        let v = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created,
                            "model": name, "choices": [], "usage": usage_json(&usage),
                        });
                        let _ = tx.send(SseEvent::default().data(v.to_string())).await;
                    }
                    let _ = tx.send(SseEvent::default().data("[DONE]")).await;
                    return;
                }
                failed @ (Event::Error(_) | Event::Failed(_)) => {
                    let (kind, e) = failure(failed);
                    let v = json!({"error": {"message": e, "type": kind}});
                    let _ = tx.send(SseEvent::default().data(v.to_string())).await;
                    return;
                }
            }
        }
    });
    Ok(sse_response(rx))
}

/// `POST /v1/completions`.
pub(super) async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("only n = 1 is supported"));
    }
    let prompt = match req.prompt {
        Value::String(s) => s,
        Value::Array(items) if items.len() == 1 && items[0].is_string() => items[0].as_str().unwrap_or("").to_string(),
        _ => return Err(ApiError::bad_request("prompt must be a single string")),
    };
    let (name, model) = resolve(&state, &req.model).await?;
    let prepared = prepare(
        &model,
        &state.limits,
        Spec {
            prompt: &prompt,
            special: false,
            sampling: Sampling {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                min_p: req.min_p,
                repetition_penalty: req.repetition_penalty,
                presence_penalty: req.presence_penalty,
                frequency_penalty: req.frequency_penalty,
                seed: req.seed,
            },
            default_temperature: 1.0,
            max_tokens: req.max_tokens,
            stop: stop_list(req.stop)?,
            json: JsonMode::Off,
            tools: Vec::new(),
        },
    )?;
    let id = response_id("cmpl-");
    let created = unix_now();

    if !req.stream {
        let out = collect(&model, prepared.request).await?;
        return Ok(Json(json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": name,
            "choices": [{
                "index": 0,
                "text": out.text,
                "logprobs": null,
                "finish_reason": finish_reason(&out.reason, false),
            }],
            "usage": usage_json(&out.usage),
        }))
        .into_response());
    }

    let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
    let mut events = submit(&model, prepared.request)?;
    let (tx, rx) = mpsc::channel::<SseEvent>(64);
    tokio::spawn(async move {
        let chunk = |text: &str, finish: Option<&str>| {
            let v = json!({
                "id": id, "object": "text_completion", "created": created, "model": name,
                "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}],
            });
            SseEvent::default().data(v.to_string())
        };
        while let Some(event) = events.recv().await {
            match event {
                Event::Text(t) => {
                    if tx.send(chunk(&t, None)).await.is_err() {
                        return;
                    }
                }
                Event::Done { reason, usage } => {
                    let _ = tx.send(chunk("", Some(finish_reason(&reason, false)))).await;
                    if include_usage {
                        let v = json!({
                            "id": id, "object": "text_completion", "created": created,
                            "model": name, "choices": [], "usage": usage_json(&usage),
                        });
                        let _ = tx.send(SseEvent::default().data(v.to_string())).await;
                    }
                    let _ = tx.send(SseEvent::default().data("[DONE]")).await;
                    return;
                }
                failed @ (Event::Error(_) | Event::Failed(_)) => {
                    let (kind, e) = failure(failed);
                    let v = json!({"error": {"message": e, "type": kind}});
                    let _ = tx.send(SseEvent::default().data(v.to_string())).await;
                    return;
                }
            }
        }
    });
    Ok(sse_response(rx))
}
