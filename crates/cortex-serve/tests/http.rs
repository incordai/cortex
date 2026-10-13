//! The server end to end over a real socket, with a scripted byte-level
//! model on the CPU backend: every protocol, streaming and not, stop
//! handling, JSON mode, batching, the prompt cache and model unloading.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::PrefixState;
use cortex_serve::{
    AppState, ChatMessage, ChatTemplate, EngineConfig, Limits, Registry, Role, ServeModel, ServedModel, ToolSpec,
    Tokenizer, spawn,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const EOS: u32 = 256;
const VOCAB: usize = 257;
const REPLY: &[u8] = b"hello world";
/// What the model says when the conversation contains `TOOLTEST`.
const TOOL_REPLY: &[u8] = b"Let me look.<tool_call>{\"name\": \"lookup\", \"arguments\": {\"q\": \"cortex\", \"n\": 3}}</tool_call>";

/// Bytes are tokens; 256 ends a turn.
struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, _special: bool) -> Result<Vec<u32>, String> {
        Ok(text.bytes().map(u32::from).collect())
    }
    fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().filter(|&&t| t < 256).map(|&t| t as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
    fn stop_tokens(&self) -> Vec<u32> {
        vec![EOS]
    }
    fn token_bytes(&self, token: u32) -> Option<Vec<u8>> {
        (token < 256).then(|| vec![token as u8])
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

struct Template;

impl ChatTemplate for Template {
    fn render(&self, messages: &[ChatMessage], _tools: &[ToolSpec]) -> Result<String, String> {
        let mut out = String::new();
        for m in messages {
            let tag = match m.role {
                Role::System => "S",
                Role::User => "U",
                Role::Assistant => "A",
                Role::Tool => "T",
            };
            out.push_str(&format!("{tag}:{}\n", m.content));
        }
        out.push_str("A:");
        Ok(out)
    }
}

/// The tokens a sequence has seen.
#[derive(Clone, Debug, Default)]
struct History(Vec<u32>);

impl PrefixState for History {
    fn trim_to(&mut self, len: usize) -> bool {
        if self.0.len() < len {
            return false;
        }
        self.0.truncate(len);
        true
    }
}

/// After the last `A:` it says `hello world` (or, given `TOOLTEST`, a tool
/// call) and stops. When the history contains `JSONTEST` it instead follows
/// a fixed preference whose favourite byte (0x01) is never valid JSON, so
/// only a constraint makes it emit JSON.
struct ScriptedModel {
    largest_batch: Arc<AtomicUsize>,
    step_delay: Duration,
}

fn contains(hay: &[u32], needle: &[u8]) -> bool {
    hay.windows(needle.len())
        .any(|w| w.iter().zip(needle).all(|(a, b)| *a == u32::from(*b)))
}

fn next_logits(history: &[u32]) -> Vec<f32> {
    // A model fault: the engine must survive it.
    assert!(!contains(history, b"CRASH"), "the model crashed on purpose");
    let mut row = vec![-1.0f32; VOCAB];
    if contains(history, b"JSONTEST") {
        for (token, weight) in [(1u32, 9.0), (b'"' as u32, 8.0), (b':' as u32, 7.0), (b'}' as u32, 5.0), (EOS, 4.0), (b'{' as u32, 3.0)] {
            row[token as usize] = weight;
        }
        return row;
    }
    let start = history
        .windows(2)
        .rposition(|w| w == [u32::from(b'A'), u32::from(b':')])
        .map_or(history.len(), |i| i + 2);
    let said = history.len() - start;
    let script = if contains(history, b"TOOLTEST") { TOOL_REPLY } else { REPLY };
    let target = script.get(said).map_or(EOS, |&b| u32::from(b));
    row[target as usize] = 10.0;
    row
}

impl ServeModel for ScriptedModel {
    type State = History;

    fn new_state(&self) -> History {
        History::default()
    }

    fn forward(&mut self, state: &mut History, tokens: &[u32]) -> Tensor<2> {
        let mut rows = Vec::with_capacity(tokens.len() * VOCAB);
        for &t in tokens {
            state.0.push(t);
            rows.extend(next_logits(&state.0));
        }
        Tensor::from_data(TensorData::new(rows, [tokens.len(), VOCAB]), &Device::default())
    }

    fn forward_batch(&mut self, states: &mut [&mut History], tokens: &[u32]) -> Tensor<2> {
        self.largest_batch.fetch_max(states.len(), Ordering::SeqCst);
        std::thread::sleep(self.step_delay);
        let mut rows = Vec::with_capacity(tokens.len() * VOCAB);
        for (state, &t) in states.iter_mut().zip(tokens) {
            state.0.push(t);
            rows.extend(next_logits(&state.0));
        }
        Tensor::from_data(TensorData::new(rows, [tokens.len(), VOCAB]), &Device::default())
    }

    fn state_bytes(&self, state: &History) -> usize {
        state.0.len() * 4
    }
}

fn scripted(largest_batch: Arc<AtomicUsize>, config: EngineConfig) -> ServedModel {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(ByteTokenizer);
    let engine = spawn(
        move || ScriptedModel {
            largest_batch: largest_batch.clone(),
            step_delay: Duration::from_millis(5),
        },
        tokenizer.clone(),
        config,
    );
    ServedModel::new(engine, tokenizer, Arc::new(Template))
}

struct Server {
    addr: std::net::SocketAddr,
    largest_batch: Arc<AtomicUsize>,
}

async fn start() -> Server {
    let largest_batch = Arc::new(AtomicUsize::new(0));
    let registry = Registry::new(2);
    registry.add_loaded(
        "toy",
        scripted(
            largest_batch.clone(),
            EngineConfig {
                max_batch: 8,
                prefill_chunk: 4,
                prompt_cache_bytes: 1 << 20,
            },
        ),
    );
    let state = Arc::new(AppState {
        registry,
        limits: Limits::default(),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { cortex_serve::serve(listener, state).await.unwrap() });
    Server { addr, largest_batch }
}

/// A minimal HTTP/1.1 client: returns the status and the (de-chunked) body.
async fn request(addr: std::net::SocketAddr, method: &str, path: &str, body: Option<Value>) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(payload.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(60), stream.read_to_end(&mut raw))
        .await
        .expect("the server answered in time")
        .unwrap();
    let text = String::from_utf8(raw).unwrap();
    let (headers, body) = text.split_once("\r\n\r\n").expect("a header block");
    let status: u16 = headers.split(' ').nth(1).unwrap().parse().unwrap();
    let body = if headers.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(body)
    } else {
        body.to_string()
    };
    (status, body)
}

fn dechunk(mut body: &str) -> String {
    let mut out = String::new();
    loop {
        let (size, rest) = body.split_once("\r\n").expect("a chunk size line");
        let size = usize::from_str_radix(size.trim(), 16).unwrap();
        if size == 0 {
            return out;
        }
        out.push_str(&rest[..size]);
        body = &rest[size + 2..];
    }
}

/// The `data:` payloads of an SSE body, with their event names.
fn sse(body: &str) -> Vec<(Option<String>, String)> {
    body.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .filter_map(|block| {
            let mut event = None;
            let mut data = None;
            for line in block.lines() {
                if let Some(e) = line.strip_prefix("event: ").or_else(|| line.strip_prefix("event:")) {
                    event = Some(e.trim().to_string());
                }
                if let Some(d) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) {
                    data = Some(d.trim().to_string());
                }
            }
            data.map(|d| (event, d))
        })
        .collect()
}

fn chat_body(extra: Value) -> Value {
    let mut body = json!({
        "model": "toy",
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        "temperature": 0,
    });
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_chat_returns_the_reply_and_honours_limits() {
    let server = start().await;
    let (status, body) = request(server.addr, "POST", "/v1/chat/completions", Some(chat_body(json!({})))).await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "hello world");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], "S:be brief\nU:hi\nA:".len());

    let (_, body) = request(server.addr, "POST", "/v1/chat/completions", Some(chat_body(json!({"max_tokens": 5})))).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((v["choices"][0]["message"]["content"].as_str(), v["choices"][0]["finish_reason"].as_str()), (Some("hello"), Some("length")));

    let (_, body) = request(server.addr, "POST", "/v1/chat/completions", Some(chat_body(json!({"stop": ["wor"]})))).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "hello ", "the stop string is cut, not emitted");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    let (status, body) = request(server.addr, "POST", "/v1/chat/completions", Some(chat_body(json!({"temperature": 9})))).await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_streams_chunks_then_done_with_usage() {
    let server = start().await;
    let body = chat_body(json!({"stream": true, "stream_options": {"include_usage": true}}));
    let (status, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, 200, "{raw}");
    let events = sse(&raw);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1].iter().map(|(_, d)| serde_json::from_str(d).unwrap()).collect();
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"].get(0).and_then(|ch| ch["delta"]["content"].as_str()).map(str::to_string))
        .collect();
    assert_eq!(text, "hello world");
    assert!(chunks.iter().any(|c| c["choices"].get(0).is_some_and(|ch| ch["finish_reason"] == "stop")));
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], REPLY.len() + 1, "the reply plus the stop token");
}

#[tokio::test(flavor = "multi_thread")]
async fn completions_and_models_endpoints() {
    let server = start().await;
    let (_, body) = request(
        server.addr,
        "POST",
        "/v1/completions",
        Some(json!({"model": "toy", "prompt": "A:", "temperature": 0, "max_tokens": 4})),
    )
    .await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["text"], "hell");
    let (_, body) = request(server.addr, "GET", "/v1/models", None).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "toy");
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_messages_plain_and_streamed() {
    let server = start().await;
    let body = json!({
        "model": "toy", "max_tokens": 64, "temperature": 0, "system": "be brief",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let (status, raw) = request(server.addr, "POST", "/v1/messages", Some(body.clone())).await;
    assert_eq!(status, 200, "{raw}");
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["content"][0]["text"], "hello world");
    assert_eq!(v["stop_reason"], "end_turn");

    let mut streamed = body;
    streamed["stream"] = json!(true);
    let (_, raw) = request(server.addr, "POST", "/v1/messages", Some(streamed)).await;
    let events = sse(&raw);
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_deref().unwrap_or("")).collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(&names[names.len() - 3..], ["content_block_stop", "message_delta", "message_stop"]);
    let text: String = events
        .iter()
        .filter(|(e, _)| e.as_deref() == Some("content_block_delta"))
        .map(|(_, d)| serde_json::from_str::<Value>(d).unwrap()["delta"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(text, "hello world");
}

#[tokio::test(flavor = "multi_thread")]
async fn ollama_generate_streams_ndjson_and_chat_answers_whole() {
    let server = start().await;
    let (status, raw) = request(
        server.addr,
        "POST",
        "/api/generate",
        Some(json!({"model": "toy", "prompt": "hi", "options": {"temperature": 0}})),
    )
    .await;
    assert_eq!(status, 200, "{raw}");
    let lines: Vec<Value> = raw.lines().filter(|l| !l.is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect();
    let text: String = lines.iter().map(|l| l["response"].as_str().unwrap_or("")).collect();
    assert_eq!(text, "hello world");
    assert_eq!(lines.last().unwrap()["done"], true);
    assert_eq!(lines.last().unwrap()["done_reason"], "stop");

    let (_, raw) = request(
        server.addr,
        "POST",
        "/api/chat",
        Some(json!({"model": "toy", "stream": false, "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": "hi"}]})),
    )
    .await;
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["message"]["content"], "hello world");
    let (_, raw) = request(server.addr, "GET", "/api/tags", None).await;
    assert_eq!(serde_json::from_str::<Value>(&raw).unwrap()["models"][0]["name"], "toy");
}

#[tokio::test(flavor = "multi_thread")]
async fn json_mode_turns_garbage_into_a_json_object() {
    let server = start().await;
    let ask = |extra: Value| {
        let mut body = json!({
            "model": "toy", "temperature": 0, "max_tokens": 16,
            "messages": [{"role": "user", "content": "JSONTEST"}],
        });
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    };
    let (_, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(ask(json!({})))).await;
    let v: Value = serde_json::from_str(&raw).unwrap();
    let free = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    assert!(serde_json::from_str::<Value>(&free).is_err(), "unconstrained output is not JSON: {free:?}");

    let (_, raw) = request(
        server.addr,
        "POST",
        "/v1/chat/completions",
        Some(ask(json!({"response_format": {"type": "json_object"}}))),
    )
    .await;
    let v: Value = serde_json::from_str(&raw).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(content).expect("constrained output is JSON");
    assert!(parsed.is_object(), "{content}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "it ended because the object closed");
}

#[tokio::test(flavor = "multi_thread")]
async fn json_schema_shapes_the_output_exactly() {
    let server = start().await;
    let schema = json!({
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": false,
    });
    let body = json!({
        "model": "toy", "temperature": 0, "max_tokens": 32,
        "messages": [{"role": "user", "content": "JSONTEST"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}},
    });
    let (status, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, 200, "{raw}");
    let v: Value = serde_json::from_str(&raw).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(content).expect("JSON");
    let obj = parsed.as_object().expect("an object");
    assert_eq!(obj.keys().collect::<Vec<_>>(), ["ok"], "exactly the schema's members: {content}");
    assert!(obj["ok"].is_boolean(), "{content}");

    // Ollama's `format` takes the same schema; unenforceable ones are refused.
    let (status, raw) = request(
        server.addr,
        "POST",
        "/api/generate",
        Some(json!({"model": "toy", "prompt": "JSONTEST", "stream": false, "options": {"temperature": 0, "num_predict": 32}, "format": schema})),
    )
    .await;
    assert_eq!(status, 200, "{raw}");
    let v: Value = serde_json::from_str(&raw).unwrap();
    let parsed: Value = serde_json::from_str(v["response"].as_str().unwrap()).expect("JSON");
    assert!(parsed["ok"].is_boolean(), "{parsed}");
    let bad = json!({"model": "toy", "prompt": "x", "stream": false, "format": {"type": "string", "pattern": "^a"}});
    let (status, raw) = request(server.addr, "POST", "/api/generate", Some(bad)).await;
    assert_eq!(status, 400, "{raw}");
    assert!(raw.contains("pattern"), "{raw}");
}

fn lookup_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}, "required": ["q"]},
    }})
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_calls_stream_as_they_are_written() {
    let server = start().await;
    let messages = json!([{"role": "user", "content": "TOOLTEST"}]);

    // OpenAI: the call opens with its name, then its arguments arrive in
    // several deltas that concatenate to the arguments' JSON.
    let body = json!({"model": "toy", "temperature": 0, "stream": true, "messages": messages, "tools": [lookup_tool()]});
    let (status, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, 200, "{raw}");
    let chunks: Vec<Value> = sse(&raw)
        .into_iter()
        .filter(|(_, d)| d != "[DONE]")
        .map(|(_, d)| serde_json::from_str(&d).unwrap())
        .collect();
    let deltas: Vec<&Value> = chunks.iter().filter_map(|c| c["choices"].get(0).map(|ch| &ch["delta"])).collect();
    let text: String = deltas.iter().filter_map(|d| d["content"].as_str()).collect();
    assert_eq!(text, "Let me look.");
    let call_parts: Vec<&Value> = deltas.iter().filter_map(|d| d["tool_calls"].get(0)).collect();
    assert_eq!(call_parts[0]["function"]["name"], "lookup");
    assert!(call_parts[0]["id"].as_str().is_some_and(|id| id.starts_with("call_")));
    let arg_pieces: Vec<&str> = call_parts.iter().filter_map(|p| p["function"]["arguments"].as_str()).filter(|s| !s.is_empty()).collect();
    assert!(arg_pieces.len() > 1, "the arguments streamed in pieces, not whole: {arg_pieces:?}");
    let args: Value = serde_json::from_str(&arg_pieces.concat()).unwrap();
    assert_eq!(args, json!({"q": "cortex", "n": 3}));
    assert!(chunks.iter().any(|c| c["choices"].get(0).is_some_and(|ch| ch["finish_reason"] == "tool_calls")));

    // Anthropic: a tool_use block with input_json_delta pieces.
    let body = json!({"model": "toy", "max_tokens": 200, "temperature": 0, "stream": true, "messages": messages,
                      "tools": [{"name": "lookup", "input_schema": lookup_tool()["function"]["parameters"]}]});
    let (_, raw) = request(server.addr, "POST", "/v1/messages", Some(body)).await;
    let events: Vec<(String, Value)> = sse(&raw)
        .into_iter()
        .map(|(e, d)| (e.unwrap_or_default(), serde_json::from_str(&d).unwrap()))
        .collect();
    let tool_start = events
        .iter()
        .find(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "tool_use")
        .expect("a tool_use block");
    assert_eq!(tool_start.1["content_block"]["name"], "lookup");
    let json_pieces: Vec<&str> = events
        .iter()
        .filter(|(_, d)| d["delta"]["type"] == "input_json_delta")
        .map(|(_, d)| d["delta"]["partial_json"].as_str().unwrap())
        .collect();
    assert!(json_pieces.len() > 1, "{json_pieces:?}");
    assert_eq!(serde_json::from_str::<Value>(&json_pieces.concat()).unwrap(), json!({"q": "cortex", "n": 3}));
    let stop = events.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(stop.1["delta"]["stop_reason"], "tool_use");

    // Ollama: the whole call in its own line, before the final line.
    let body = json!({"model": "toy", "options": {"temperature": 0}, "messages": messages, "tools": [lookup_tool()]});
    let (_, raw) = request(server.addr, "POST", "/api/chat", Some(body)).await;
    let lines: Vec<Value> = raw.lines().filter(|l| !l.is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect();
    let call_line = lines.iter().position(|l| l["message"]["tool_calls"].is_array()).expect("a tool_calls line");
    assert!(call_line < lines.len() - 1, "sent before the final line");
    assert_eq!(lines[call_line]["message"]["tool_calls"][0]["function"]["arguments"], json!({"q": "cortex", "n": 3}));
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_tool_choice_makes_a_valid_call() {
    let server = start().await;
    let body = |choice: Value| {
        // The toy model spells keys as \u escapes and pads with whitespace
        // where it may, so a valid call takes a few hundred tokens.
        json!({"model": "toy", "temperature": 0, "max_tokens": 600, "tools": [lookup_tool()], "tool_choice": choice,
               "messages": [{"role": "user", "content": "JSONTEST"}]})
    };
    // Left to itself the model writes no call.
    let (_, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(body(json!("auto")))).await;
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert!(v["choices"][0]["message"]["tool_calls"].is_null(), "{raw}");

    for choice in [json!("required"), json!({"type": "function", "function": {"name": "lookup"}})] {
        let (status, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(body(choice))).await;
        assert_eq!(status, 200, "{raw}");
        let v: Value = serde_json::from_str(&raw).unwrap();
        let call = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "lookup", "{raw}");
        let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert!(args["q"].is_string(), "the required argument is there: {args}");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    }
    let (status, _) = request(
        server.addr,
        "POST",
        "/v1/chat/completions",
        Some(body(json!({"type": "function", "function": {"name": "nope"}}))),
    )
    .await;
    assert_eq!(status, 400, "an unknown forced tool is refused");
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_share_decode_steps_and_the_prompt_cache_hits() {
    let server = start().await;
    let calls = (0..4).map(|i| {
        let addr = server.addr;
        tokio::spawn(async move {
            let body = json!({"model": "toy", "temperature": 0, "messages": [{"role": "user", "content": format!("q{i}")}]});
            request(addr, "POST", "/v1/chat/completions", Some(body)).await
        })
    });
    for call in calls.collect::<Vec<_>>() {
        let (status, body) = call.await.unwrap();
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "hello world");
    }
    assert!(
        server.largest_batch.load(Ordering::SeqCst) > 1,
        "concurrent requests were decoded one at a time"
    );

    // The same conversation again resumes from the cached prompt.
    let body = chat_body(json!({}));
    request(server.addr, "POST", "/v1/chat/completions", Some(body.clone())).await;
    let (_, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(body)).await;
    let v: Value = serde_json::from_str(&raw).unwrap();
    let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(cached > 0, "the repeated prompt was prefilled from scratch");
    assert_eq!(v["choices"][0]["message"]["content"], "hello world", "a cached prefix gives the same answer");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_crash_fails_its_requests_and_the_engine_recovers() {
    let server = start().await;
    let ask = |content: &str| json!({"model": "toy", "temperature": 0, "messages": [{"role": "user", "content": content}]});
    let (status, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(ask("CRASH now"))).await;
    assert_eq!(status, 500, "a model fault is a server error: {raw}");
    assert!(raw.contains("crashed on purpose"), "{raw}");
    // The next request is served by a rebuilt model.
    let (status, raw) = request(server.addr, "POST", "/v1/chat/completions", Some(ask("hi"))).await;
    assert_eq!(status, 200, "{raw}");
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "hello world");
}

#[test]
fn an_engine_that_keeps_crashing_is_retired_and_reloaded() {
    let registry = Registry::new(2);
    let loads = Arc::new(AtomicUsize::new(0));
    let counter = loads.clone();
    registry.add_lazy(
        "fragile",
        Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            let tokenizer: Arc<dyn Tokenizer> = Arc::new(ByteTokenizer);
            // Every build of this model fails.
            let engine = spawn(
                || -> ScriptedModel { panic!("cannot build") },
                tokenizer.clone(),
                EngineConfig::default(),
            );
            Ok(ServedModel::new(engine, tokenizer, Arc::new(Template)))
        }),
    );
    let (_, model) = registry.get("fragile").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while model.engine.is_alive() {
        assert!(std::time::Instant::now() < deadline, "the crash-looping engine never stopped");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    registry.get("fragile").unwrap();
    assert_eq!(loads.load(Ordering::SeqCst), 2, "a dead engine is loaded again on the next request");
}

#[test]
fn loading_one_model_never_blocks_another_and_loads_are_shared() {
    let registry = Arc::new(Registry::new(4));
    registry.add_loaded("ready", scripted(Arc::new(AtomicUsize::new(0)), EngineConfig::default()));
    let loads = Arc::new(AtomicUsize::new(0));
    let counter = loads.clone();
    registry.add_lazy(
        "slow",
        Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(600));
            Ok(scripted(Arc::new(AtomicUsize::new(0)), EngineConfig::default()))
        }),
    );
    let waiters: Vec<_> = (0..4)
        .map(|_| {
            let r = registry.clone();
            std::thread::spawn(move || r.get("slow").map(|(name, _)| name))
        })
        .collect();
    std::thread::sleep(Duration::from_millis(100));
    let t = std::time::Instant::now();
    registry.get("ready").unwrap();
    assert!(
        t.elapsed() < Duration::from_millis(200),
        "a resident model waited {:?} behind another model's load",
        t.elapsed()
    );
    for w in waiters {
        assert_eq!(w.join().unwrap().unwrap(), "slow");
    }
    assert_eq!(loads.load(Ordering::SeqCst), 1, "concurrent requests share one load");

    // A failing load reports its error to every request that waited for it.
    let failing = Arc::new(AtomicUsize::new(0));
    let counter = failing.clone();
    registry.add_lazy(
        "broken",
        Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(300));
            Err("no weights".into())
        }),
    );
    let waiters: Vec<_> = (0..3)
        .map(|_| {
            let r = registry.clone();
            std::thread::spawn(move || r.get("broken").map(|_| ()))
        })
        .collect();
    for w in waiters {
        assert!(matches!(w.join().unwrap(), Err(cortex_serve::registry::RegistryError::LoadFailed(e)) if e == "no weights"));
    }
    assert_eq!(failing.load(Ordering::SeqCst), 1, "the waiters shared the failed load");
}

#[test]
fn the_least_recently_used_model_is_unloaded() {
    let registry = Registry::new(1);
    let loads = Arc::new(AtomicUsize::new(0));
    for name in ["a", "b"] {
        let loads = loads.clone();
        registry.add_lazy(
            name,
            Box::new(move || {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(scripted(Arc::new(AtomicUsize::new(0)), EngineConfig::default()))
            }),
        );
    }
    registry.get("a").unwrap();
    assert_eq!(registry.resident(), ["a"]);
    registry.get("b").unwrap();
    assert_eq!(registry.resident(), ["b"], "a was unloaded to make room");
    registry.get("b").unwrap();
    assert_eq!(loads.load(Ordering::SeqCst), 2, "a resident model is not reloaded");
    assert!(registry.get("missing").is_err());
}

/// A real model behind the engine, recording the largest batch it was
/// asked to decode.
struct CountedLlama {
    model: cortex_llm::Llama,
    largest_batch: Arc<AtomicUsize>,
}

impl ServeModel for CountedLlama {
    type State = cortex_llm::LlamaState;

    fn new_state(&self) -> Self::State {
        self.model.new_state()
    }

    fn forward(&mut self, state: &mut Self::State, tokens: &[u32]) -> Tensor<2> {
        ServeModel::forward(&mut self.model, state, tokens)
    }

    fn forward_batch(&mut self, states: &mut [&mut Self::State], tokens: &[u32]) -> Tensor<2> {
        self.largest_batch.fetch_max(states.len(), Ordering::SeqCst);
        ServeModel::forward_batch(&mut self.model, states, tokens)
    }

    fn state_bytes(&self, state: &Self::State) -> usize {
        ServeModel::state_bytes(&self.model, state)
    }
}

/// A Llama checkpoint served for real: requests decoded together in one
/// batched step answer exactly what each answers alone.
#[tokio::test(flavor = "multi_thread")]
async fn a_served_llama_batches_concurrent_requests_without_changing_their_answers() {
    let dir = common::scratch("llama");
    common::write_llama(&dir, VOCAB);
    let model = cortex_llm::Llama::load(&dir, &Device::default()).unwrap();

    let largest_batch = Arc::new(AtomicUsize::new(0));
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(ByteTokenizer);
    let counter = largest_batch.clone();
    let engine = spawn(
        move || CountedLlama {
            model: model.clone(),
            largest_batch: counter.clone(),
        },
        tokenizer.clone(),
        EngineConfig {
            max_batch: 8,
            prefill_chunk: 16,
            prompt_cache_bytes: 0,
        },
    );
    let registry = Registry::new(1);
    registry.add_loaded("llama", ServedModel::new(engine, tokenizer, Arc::new(Template)));
    let state = Arc::new(AppState {
        registry,
        limits: Limits::default(),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { cortex_serve::serve(listener, state).await.unwrap() });

    let ask = |i: usize| {
        json!({"model": "llama", "temperature": 0, "max_tokens": 12,
               "messages": [{"role": "user", "content": format!("question number {i}")}]})
    };
    let answer = |raw: &str| -> (String, u64) {
        let v: Value = serde_json::from_str(raw).unwrap();
        let text = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
        (text, v["usage"]["completion_tokens"].as_u64().unwrap())
    };

    let mut alone = Vec::new();
    for i in 0..4 {
        let (status, raw) = request(addr, "POST", "/v1/chat/completions", Some(ask(i))).await;
        assert_eq!(status, 200, "{raw}");
        alone.push(answer(&raw));
    }
    assert_eq!(largest_batch.load(Ordering::SeqCst), 1, "one request at a time decodes alone");
    assert!(alone.iter().any(|(_, n)| *n > 1), "the model stopped at once: {alone:?}");

    let calls: Vec<_> = (0..4)
        .map(|i| tokio::spawn(async move { request(addr, "POST", "/v1/chat/completions", Some(ask(i))).await }))
        .collect();
    for (i, call) in calls.into_iter().enumerate() {
        let (status, raw) = call.await.unwrap();
        assert_eq!(status, 200, "{raw}");
        assert_eq!(answer(&raw), alone[i], "request {i} answered differently in a batch");
    }
    assert!(
        largest_batch.load(Ordering::SeqCst) > 1,
        "concurrent requests were decoded one at a time"
    );
    let _ = std::fs::remove_dir_all(dir);
}
