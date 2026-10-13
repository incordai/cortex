//! Anthropic-compatible `/v1/messages`.
//!
//! Content may be a string or a list of blocks (`text`, `tool_use`,
//! `tool_result`). Streaming emits the named event sequence
//! `message_start`, `content_block_start`, `content_block_delta`*,
//! `content_block_stop`, `message_delta`, `message_stop`. A tool call opens
//! its `tool_use` block as soon as its name is written and streams its input
//! as `input_json_delta`s; `tool_choice` `any` or `tool` forces a valid call.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::Event as SseEvent;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::{
    ApiError, AppState, JsonMode, Sampling, Spec, ToolChoice, apply_tool_choice, collect, failure, prepare, render,
    resolve, response_id, sse_response, submit, tool_ids, tool_names, tool_parser,
};
use crate::engine::{Event, FinishReason};
use crate::model::{ChatMessage, Role, ToolCall, ToolSpec};
use crate::tools::{ToolEvent, parse_tool_calls};

#[derive(Deserialize)]
struct Message {
    role: String,
    content: Value,
}

#[derive(Deserialize)]
struct Tool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    input_schema: Value,
}

#[derive(Deserialize)]
pub(super) struct MessagesRequest {
    #[serde(default)]
    model: String,
    max_tokens: usize,
    #[serde(default)]
    system: Value,
    messages: Vec<Message>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<Tool>,
    tool_choice: Option<Value>,
}

fn block_text(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    out.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""));
                } else {
                    return Err(ApiError::bad_request("only text blocks are supported here"));
                }
            }
            Ok(out)
        }
        Value::Null => Ok(String::new()),
        _ => Err(ApiError::bad_request("content must be a string or a list of blocks")),
    }
}

/// Converts the conversation, splitting `tool_result` blocks into tool
/// messages and `tool_use` blocks into assistant tool calls.
fn conversation(system: &Value, raw: Vec<Message>) -> Result<Vec<ChatMessage>, ApiError> {
    let mut out = Vec::new();
    let system = block_text(system)?;
    if !system.is_empty() {
        out.push(ChatMessage::text(Role::System, system));
    }
    for m in raw {
        let role = match m.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(ApiError::bad_request(format!("unknown message role '{other}'"))),
        };
        let blocks = match m.content {
            Value::String(s) => {
                out.push(ChatMessage::text(role, s));
                continue;
            }
            Value::Array(blocks) => blocks,
            _ => return Err(ApiError::bad_request("content must be a string or a list of blocks")),
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        for b in blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                Some("tool_use") => calls.push(ToolCall {
                    id: b.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                    name: b.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                    arguments: b.get("input").cloned().unwrap_or(Value::Null),
                }),
                Some("tool_result") => {
                    let mut result = ChatMessage::text(Role::Tool, block_text(b.get("content").unwrap_or(&Value::Null))?);
                    result.tool_call_id = b.get("tool_use_id").and_then(Value::as_str).map(str::to_string);
                    out.push(result);
                }
                Some(other) => return Err(ApiError::bad_request(format!("content blocks of type '{other}' are not supported"))),
                None => return Err(ApiError::bad_request("a content block has no type")),
            }
        }
        if !text.is_empty() || !calls.is_empty() {
            out.push(ChatMessage {
                role,
                content: text,
                tool_calls: calls,
                tool_call_id: None,
            });
        }
    }
    Ok(out)
}

fn stop_reason(reason: &FinishReason, tool_use: bool) -> (&'static str, Value) {
    if tool_use {
        return ("tool_use", Value::Null);
    }
    match reason {
        FinishReason::Stop => ("end_turn", Value::Null),
        FinishReason::StopString(s) => ("stop_sequence", Value::String(s.clone())),
        FinishReason::Length => ("max_tokens", Value::Null),
    }
}

fn tool_choice(choice: Option<&Value>) -> Result<ToolChoice, ApiError> {
    let Some(choice) = choice else {
        return Ok(ToolChoice::Auto);
    };
    match choice.get("type").and_then(Value::as_str) {
        None | Some("auto") => Ok(ToolChoice::Auto),
        Some("none") => Ok(ToolChoice::None),
        Some("any") => Ok(ToolChoice::Required),
        Some("tool") => choice
            .get("name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string()))
            .ok_or_else(|| ApiError::bad_request("tool_choice of type 'tool' needs a name")),
        Some(other) => Err(ApiError::bad_request(format!("tool_choice type '{other}' is not supported"))),
    }
}

/// Emits the named Anthropic stream events for one tool-parser event,
/// opening and closing content blocks as needed.
struct Blocks {
    index: usize,
    text_open: bool,
}

impl Blocks {
    fn events(&mut self, event: ToolEvent) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        match event {
            ToolEvent::Text(t) => {
                if t.is_empty() {
                    return out;
                }
                if !self.text_open {
                    self.text_open = true;
                    out.push((
                        "content_block_start",
                        json!({"type": "content_block_start", "index": self.index, "content_block": {"type": "text", "text": ""}}),
                    ));
                }
                out.push((
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": self.index, "delta": {"type": "text_delta", "text": t}}),
                ));
            }
            ToolEvent::CallStart { id, name, .. } => {
                out.extend(self.close_text());
                out.push((
                    "content_block_start",
                    json!({
                        "type": "content_block_start", "index": self.index,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}},
                    }),
                ));
            }
            ToolEvent::CallArgs { delta, .. } => out.push((
                "content_block_delta",
                json!({"type": "content_block_delta", "index": self.index, "delta": {"type": "input_json_delta", "partial_json": delta}}),
            )),
            ToolEvent::CallEnd { .. } => {
                out.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
                self.index += 1;
            }
        }
        out
    }

    fn close_text(&mut self) -> Option<(&'static str, Value)> {
        if !self.text_open {
            return None;
        }
        self.text_open = false;
        let stop = json!({"type": "content_block_stop", "index": self.index});
        self.index += 1;
        Some(("content_block_stop", stop))
    }
}

fn content_blocks(text: &str, calls: &[ToolCall]) -> Vec<Value> {
    let mut blocks = Vec::new();
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    for c in calls {
        blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}));
    }
    blocks
}

/// `POST /v1/messages`.
pub(super) async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, ApiError> {
    let choice = tool_choice(req.tool_choice.as_ref())?;
    let offered: Vec<ToolSpec> = req
        .tools
        .into_iter()
        .map(|t| ToolSpec {
            name: t.name,
            description: t.description,
            parameters: t.input_schema,
        })
        .collect();
    let (tools, json) = apply_tool_choice(offered, &choice, JsonMode::Off)?;
    let (name, model) = resolve(&state, &req.model).await?;
    let chat = conversation(&req.system, req.messages)?;
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
                ..Sampling::default()
            },
            default_temperature: 1.0,
            max_tokens: Some(req.max_tokens),
            stop: req.stop_sequences,
            json,
            tools: tool_names(&tools),
        },
    )?;
    let id = response_id("msg_");
    let input_tokens = prepared.request.prompt.len();

    if !req.stream {
        let names = prepared.tools;
        let out = collect(&model, prepared.request).await?;
        let (text, calls) = if names.is_empty() {
            (out.text, Vec::new())
        } else {
            parse_tool_calls(&out.text, &names, tool_ids())
        };
        let (reason, sequence) = stop_reason(&out.reason, !calls.is_empty());
        return Ok(Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": name,
            "content": content_blocks(&text, &calls),
            "stop_reason": reason,
            "stop_sequence": sequence,
            "usage": {
                "input_tokens": out.usage.prompt_tokens,
                "output_tokens": out.usage.completion_tokens,
                "cache_read_input_tokens": out.usage.cached_tokens,
            },
        }))
        .into_response());
    }

    let mut parser = tool_parser(&prepared.tools);
    let mut events = submit(&model, prepared.request)?;
    let (tx, rx) = mpsc::channel::<SseEvent>(64);
    tokio::spawn(async move {
        let send = |name: &'static str, data: Value| SseEvent::default().event(name).data(data.to_string());
        let start = json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": name,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        });
        if tx.send(send("message_start", start)).await.is_err() {
            return;
        }
        let mut blocks = Blocks {
            index: 0,
            text_open: false,
        };
        while let Some(event) = events.recv().await {
            match event {
                Event::Text(t) => {
                    let pieces = match parser.as_mut() {
                        Some(p) => p.push(&t),
                        None => vec![ToolEvent::Text(t)],
                    };
                    for (kind, data) in pieces.into_iter().flat_map(|e| blocks.events(e)) {
                        if tx.send(send(kind, data)).await.is_err() {
                            return;
                        }
                    }
                }
                Event::Done { reason, usage } => {
                    let mut called = false;
                    if let Some(p) = parser.as_mut() {
                        for (kind, data) in p.finish().into_iter().flat_map(|e| blocks.events(e)) {
                            let _ = tx.send(send(kind, data)).await;
                        }
                        called = p.has_calls();
                    }
                    if let Some((kind, data)) = blocks.close_text() {
                        let _ = tx.send(send(kind, data)).await;
                    }
                    let (reason, sequence) = stop_reason(&reason, called);
                    let delta = json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": sequence},
                        "usage": {"output_tokens": usage.completion_tokens},
                    });
                    let _ = tx.send(send("message_delta", delta)).await;
                    let _ = tx.send(send("message_stop", json!({"type": "message_stop"}))).await;
                    return;
                }
                failed @ (Event::Error(_) | Event::Failed(_)) => {
                    let (kind, e) = failure(failed);
                    let kind = if kind == "server_error" { "api_error" } else { kind };
                    let err = json!({"type": "error", "error": {"type": kind, "message": e}});
                    let _ = tx.send(send("error", err)).await;
                    return;
                }
            }
        }
    });
    Ok(sse_response(rx))
}
