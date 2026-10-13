#![warn(missing_docs)]

//! HTTP serving for `cortex-llm` models, compatible with the OpenAI,
//! Anthropic and Ollama APIs.
//!
//! - [`model`]: what a model provides ([`ServeModel`], [`Tokenizer`],
//!   [`ChatTemplate`]).
//! - [`hf`]: a checkpoint directory served as is ([`load_model`]): its
//!   weights, `tokenizer.json` and chat template.
//! - [`engine`]: one worker per model, batching every active request
//!   (continuous batching), with chunked prefill and a prompt cache.
//! - [`registry`]: several models, loaded on demand and unloaded least
//!   recently used first.
//! - [`constrain`]: constrained decoding to JSON matching a [`Schema`]
//!   ([`schema`] compiles JSON Schema).
//! - [`tools`]: tool-call extraction from model output, whole or streamed.
//! - [`api`]: the HTTP endpoints; [`serve`] runs them.

pub mod api;
pub mod constrain;
pub mod engine;
pub mod hf;
pub mod model;
pub mod registry;
pub mod schema;
pub mod tools;

use std::sync::Arc;

pub use api::{AppState, Limits, router};
pub use engine::{EngineConfig, EngineHandle, Event, FinishReason, GenerationRequest, Usage, spawn};
pub use hf::{HfChatTemplate, HfTokenizer, load_model};
pub use model::{ChatMessage, ChatTemplate, Role, ServeModel, StreamDecoder, ToolCall, ToolSpec, Tokenizer};
pub use registry::{Registry, ServedModel};
pub use schema::{Schema, SchemaError};

/// Serves every endpoint on `listener` until the server stops.
pub async fn serve(listener: tokio::net::TcpListener, state: Arc<AppState>) -> std::io::Result<()> {
    axum::serve(listener, router(state)).await
}
