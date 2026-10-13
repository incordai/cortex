//! What a model must provide to be served: per-sequence state, a forward
//! pass over it, a tokenizer and a chat template.
//!
//! Callers: `engine.rs` drives [`ServeModel`] on its worker thread; the HTTP
//! handlers in `api/` use [`Tokenizer`] and [`ChatTemplate`] to turn requests
//! into token ids. Sibling: `cortex_llm::LanguageModel`, the single-sequence
//! contract that `cortex_llm::generate` drives.

use cortex_core::tensor::Tensor;
use cortex_llm::PrefixState;
use serde::{Deserialize, Serialize};

/// A model that can run many independent sequences.
///
/// The state of one sequence (its KV caches, recurrent state, ...) lives in
/// [`ServeModel::State`]; the engine keeps one per request and reuses them
/// across requests through the prompt cache, which is why the state must be
/// cheaply clonable and trimmable ([`PrefixState`]).
pub trait ServeModel {
    /// Everything one sequence carries between forward passes.
    type State: PrefixState;

    /// A state holding no positions.
    fn new_state(&self) -> Self::State;

    /// Feeds `tokens` after the positions `state` already holds and returns
    /// the logits predicting the token after each of them, `[n, vocab]`.
    fn forward(&mut self, state: &mut Self::State, tokens: &[u32]) -> Tensor<2>;

    /// One new token for each of several sequences: `tokens[i]` follows
    /// `states[i]`. Returns `[states.len(), vocab]`.
    ///
    /// The default runs the sequences one after another; models whose layers
    /// accept ragged batches override it to run one batched step.
    fn forward_batch(&mut self, states: &mut [&mut Self::State], tokens: &[u32]) -> Tensor<2> {
        assert_eq!(states.len(), tokens.len(), "one token per sequence");
        let rows: Vec<Tensor<2>> = states
            .iter_mut()
            .zip(tokens)
            .map(|(state, &t)| self.forward(state, &[t]))
            .collect();
        Tensor::cat(rows, 0)
    }

    /// Bytes `state` occupies, for the prompt cache's budget.
    fn state_bytes(&self, state: &Self::State) -> usize;
}

/// A Llama-family checkpoint served directly: decode steps of every active
/// request run as one batched step (`Llama::forward_batch` in
/// `cortex_llm::llama`), whose projections read each weight once for the
/// whole batch.
impl ServeModel for cortex_llm::Llama {
    type State = cortex_llm::LlamaState;

    fn new_state(&self) -> Self::State {
        cortex_llm::Llama::new_state(self)
    }

    fn forward(&mut self, state: &mut Self::State, tokens: &[u32]) -> Tensor<2> {
        cortex_llm::Llama::forward(self, state, tokens)
    }

    fn forward_batch(&mut self, states: &mut [&mut Self::State], tokens: &[u32]) -> Tensor<2> {
        cortex_llm::Llama::forward_batch(self, states, tokens)
    }

    fn state_bytes(&self, state: &Self::State) -> usize {
        cortex_llm::Llama::state_bytes(self, state)
    }
}

/// Text to token ids and back.
pub trait Tokenizer: Send + Sync {
    /// Encodes `text`. `special` allows special-token markup in it (true for
    /// rendered chat templates, false for raw user text).
    ///
    /// # Errors
    /// Text the tokenizer cannot encode, with a message for the client.
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String>;

    /// Decodes `tokens` to text, skipping special tokens.
    fn decode(&self, tokens: &[u32]) -> String;

    /// Tokens that end a turn (end-of-sequence, end-of-turn markers).
    fn stop_tokens(&self) -> Vec<u32>;

    /// The raw bytes of one token, for constrained decoding. `None` for
    /// special tokens, which a constraint never allows.
    fn token_bytes(&self, token: u32) -> Option<Vec<u8>>;

    /// Number of token ids.
    fn vocab_size(&self) -> usize;
}

/// Who wrote a chat message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions for the assistant.
    System,
    /// The user.
    User,
    /// The model.
    Assistant,
    /// The result of a tool call.
    Tool,
}

/// A tool call the assistant made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Identifier the result message refers back to.
    pub id: String,
    /// The function called.
    pub name: String,
    /// Its arguments, as a JSON value.
    pub arguments: serde_json::Value,
}

/// One message of a conversation, protocol-neutral.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    /// Author.
    pub role: Role,
    /// Text content.
    pub content: String,
    /// Tool calls made in this (assistant) message.
    pub tool_calls: Vec<ToolCall>,
    /// For a tool result: the call it answers.
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// A plain text message.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// A function the model may call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Function name.
    pub name: String,
    /// What it does.
    #[serde(default)]
    pub description: String,
    /// JSON schema of its arguments.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// Renders a conversation into the prompt text the model was trained on.
pub trait ChatTemplate: Send + Sync {
    /// The prompt for `messages`, ending with the opening of the assistant's
    /// turn. `tools` are described to the model in whatever form it expects.
    ///
    /// # Errors
    /// A conversation the template refuses (a template may require roles to
    /// alternate, for instance), with a message for the client; the HTTP
    /// handlers answer 400 with it.
    fn render(&self, messages: &[ChatMessage], tools: &[ToolSpec]) -> Result<String, String>;
}

/// Turns a stream of token ids into text without ever emitting a partial
/// UTF-8 sequence: text is released only once the bytes decoded so far form
/// complete characters.
///
/// Decodes a sliding window rather than the whole output, so each step costs
/// the length of the window, not of the generation.
#[derive(Clone, Debug, Default)]
pub struct StreamDecoder {
    tokens: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl StreamDecoder {
    /// An empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `token` and returns the text it completes (possibly empty).
    pub fn push(&mut self, tokenizer: &dyn Tokenizer, token: u32) -> String {
        self.tokens.push(token);
        let before = tokenizer.decode(&self.tokens[self.prefix..self.read]);
        let now = tokenizer.decode(&self.tokens[self.prefix..]);
        if now.len() > before.len() && !now.ends_with('\u{FFFD}') && now.starts_with(&before) {
            let new = now[before.len()..].to_string();
            self.prefix = self.read;
            self.read = self.tokens.len();
            new
        } else {
            String::new()
        }
    }

    /// Text still held back at the end (an incomplete character), decoded
    /// as-is.
    pub fn flush(&mut self, tokenizer: &dyn Tokenizer) -> String {
        let before = tokenizer.decode(&self.tokens[self.prefix..self.read]);
        let now = tokenizer.decode(&self.tokens[self.prefix..]);
        self.prefix = self.tokens.len();
        self.read = self.tokens.len();
        now.strip_prefix(before.as_str()).unwrap_or("").to_string()
    }
}
