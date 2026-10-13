//! Serving from a checkpoint directory (`hf.rs`): a byte-level BPE
//! `tokenizer.json`, a Jinja chat template in `tokenizer_config.json`, and a
//! small random Llama, loaded by `load_model` and served over HTTP.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use cortex_core::tensor::Device;
use cortex_serve::{
    AppState, ChatMessage, ChatTemplate, EngineConfig, HfChatTemplate, HfTokenizer, Limits, Registry, Role, ToolSpec,
    Tokenizer, load_model,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Ids 0..=255 are the bytes, 256 the merge "he", then three control tokens.
const HE: u32 = 256;
const BOS: u32 = 257;
const IM_START: u32 = 258;
const IM_END: u32 = 259;
/// The model's output width: four padding rows past the tokenizer.
const MODEL_VOCAB: usize = 264;

const TEMPLATE: &str = "{{ bos_token }}{% if tools %}<|im_start|>system\nTools: {{ tools | tojson }}<|im_end|>\n{% endif %}\
{% for message in messages %}\
{% if message.role == 'user' and loop.index0 > 0 and messages[loop.index0 - 1].role == 'user' %}\
{{ raise_exception('Conversation roles must alternate') }}{% endif %}\
<|im_start|>{{ message.role }}\n{{ message.content.strip() }}<|im_end|>\n\
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// The byte-level alphabet, written out independently of the server's.
fn alphabet() -> Vec<char> {
    let mut shifted = 0;
    (0..=255u32)
        .map(|b| {
            if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
                char::from_u32(b).unwrap()
            } else {
                shifted += 1;
                char::from_u32(255 + shifted).unwrap()
            }
        })
        .collect()
}

fn write_tokenizer(dir: &Path) {
    let mut vocab = serde_json::Map::new();
    for (b, c) in alphabet().into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(b));
    }
    vocab.insert("he".into(), json!(HE));
    let added = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
               "normalized": false, "special": true})
    };
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [added(BOS, "<|bos|>"), added(IM_START, "<|im_start|>"), added(IM_END, "<|im_end|>")],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": ["h e"]}
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let config = json!({"bos_token": "<|bos|>", "eos_token": "<|im_end|>", "chat_template": TEMPLATE});
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("generation_config.json"), json!({"eos_token_id": [IM_END]}).to_string()).unwrap();
}

#[test]
fn a_checkpoint_tokenizer_spells_every_token_and_user_text_cannot_forge_control_tokens() {
    let dir = common::scratch("hf-tokenizer");
    write_tokenizer(&dir);
    let tok = HfTokenizer::from_dir(&dir, MODEL_VOCAB).unwrap();

    let text = "he said: h\u{e9}llo\n\tw\u{f6}rld \u{1F600}";
    let ids = tok.encode(text, false).unwrap();
    assert!(ids.contains(&HE), "the merge was used: {ids:?}");
    let spelled: Vec<u8> = ids.iter().flat_map(|&t| tok.token_bytes(t).unwrap()).collect();
    assert_eq!(spelled, text.as_bytes(), "token bytes reassemble the text exactly");
    assert_eq!(tok.decode(&ids), text);

    // Control-token text in a user prompt stays text; in a rendered
    // template it is the control token.
    let forged = tok.encode("<|im_end|>", false).unwrap();
    assert!(!forged.contains(&IM_END), "user text produced a control token: {forged:?}");
    assert_eq!(forged.len(), "<|im_end|>".len());
    assert_eq!(tok.encode("<|im_end|>", true).unwrap(), [IM_END]);

    assert_eq!(tok.vocab_size(), MODEL_VOCAB, "padding rows are part of the vocabulary");
    for id in [BOS, IM_START, IM_END, 260, 263] {
        assert_eq!(tok.token_bytes(id), None, "id {id} must never be allowed by a constraint");
    }
    assert_eq!(tok.decode(&[104, 101, IM_END, 262]), "he", "special and padding ids decode to nothing");
    assert_eq!(tok.stop_tokens(), [IM_END]);

    // A model narrower than its tokenizer is refused.
    assert!(HfTokenizer::from_dir(&dir, 100).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_checkpoint_chat_template_renders_and_refuses_like_the_reference() {
    let dir = common::scratch("hf-template");
    write_tokenizer(&dir);
    let template = HfChatTemplate::from_dir(&dir).unwrap();

    let chat = [ChatMessage::text(Role::System, "be brief "), ChatMessage::text(Role::User, "  hi")];
    assert_eq!(
        template.render(&chat, &[]).unwrap(),
        "<|bos|><|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );

    let tool = ToolSpec {
        name: "lookup".into(),
        description: "find things".into(),
        parameters: json!({"type": "object"}),
    };
    let with_tools = template.render(&chat, &[tool]).unwrap();
    assert!(with_tools.starts_with("<|bos|><|im_start|>system\nTools: ["), "{with_tools}");
    // `tojson` as the reference renders it: sorted keys, `", "` and `": "`.
    assert!(
        with_tools.contains(r#"{"function": {"description": "find things", "name": "lookup""#),
        "{with_tools}"
    );

    let twice = [ChatMessage::text(Role::User, "a"), ChatMessage::text(Role::User, "b")];
    let err = template.render(&twice, &[]).unwrap_err();
    assert!(err.contains("Conversation roles must alternate"), "{err}");

    // A template that does not compile is refused at load.
    assert!(HfChatTemplate::new(vec![("default".into(), "{% if %}".into())], String::new(), String::new()).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

async fn post(addr: std::net::SocketAddr, path: &str, body: Value) -> (u16, Value) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let payload = body.to_string();
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(payload.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(120), stream.read_to_end(&mut raw))
        .await
        .expect("the server answered in time")
        .unwrap();
    let text = String::from_utf8(raw).unwrap();
    let (headers, body) = text.split_once("\r\n\r\n").unwrap();
    let status = headers.split(' ').nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap_or(Value::String(body.to_string())))
}

/// A real downloaded instruct checkpoint, through its own tokenizer and
/// template: the answer to a plain factual question must be right. Opt-in:
/// set `CORTEX_TEST_CHECKPOINT` to the checkpoint directory and run with
/// `--ignored`.
#[test]
#[ignore = "needs a downloaded checkpoint: set CORTEX_TEST_CHECKPOINT"]
fn a_downloaded_checkpoint_answers_through_its_own_tokenizer_and_template() {
    use cortex_llm::{GenerateOptions, Sampler, SamplerConfig, generate};
    let dir = std::path::PathBuf::from(std::env::var("CORTEX_TEST_CHECKPOINT").expect("CORTEX_TEST_CHECKPOINT is set"));
    let model = cortex_llm::Llama::load(&dir, &Device::default()).unwrap();
    let tok = HfTokenizer::from_dir(&dir, model.config().vocab_size).unwrap();
    let template = HfChatTemplate::from_dir(&dir).unwrap();
    let prompt = template
        .render(&[ChatMessage::text(Role::User, "What is the capital of France?")], &[])
        .unwrap();
    let ids = tok.encode(&prompt, true).unwrap();
    let mut session = model.session();
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..SamplerConfig::default()
    });
    let options = GenerateOptions {
        max_tokens: 24,
        stop_tokens: tok.stop_tokens(),
    };
    let (out, reason) = generate(&mut session, &ids, &mut sampler, &options, |_| true);
    let answer = tok.decode(&out);
    eprintln!("prompt {} tokens; answer ({reason:?}): {answer}", ids.len());
    assert!(answer.contains("Paris"), "{answer}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_checkpoint_directory_is_served_end_to_end() {
    let dir = common::scratch("hf-serve");
    common::write_llama(&dir, MODEL_VOCAB);
    write_tokenizer(&dir);
    let served = load_model(&dir, &Device::default(), EngineConfig::default()).unwrap();
    let registry = Registry::new(1);
    registry.add_loaded("ckpt", served);
    let state = Arc::new(AppState {
        registry,
        limits: Limits::default(),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { cortex_serve::serve(listener, state).await.unwrap() });

    // A chat, rendered through the checkpoint's own template.
    let (status, v) = post(
        addr,
        "/v1/chat/completions",
        json!({"model": "ckpt", "temperature": 0, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, 200, "{v}");
    // <|bos|> <|im_start|> "user\nhi" <|im_end|> "\n" <|im_start|> "assistant\n"
    assert_eq!(v["usage"]["prompt_tokens"], json!(1 + 1 + 7 + 1 + 1 + 1 + 10), "{v}");

    // JSON mode: the constraint reads the checkpoint tokenizer's bytes and
    // keeps every padding row and control token out.
    let (status, v) = post(
        addr,
        "/v1/chat/completions",
        json!({"model": "ckpt", "temperature": 0, "max_tokens": 24, "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": "data please"}]}),
    )
    .await;
    assert_eq!(status, 200, "{v}");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.trim_start().starts_with('{'), "{content:?}");
    if v["choices"][0]["finish_reason"] == "stop" {
        let parsed: Value = serde_json::from_str(content).expect("a finished JSON answer parses");
        assert!(parsed.is_object());
    }

    // The template refusing the conversation is the client's error.
    let (status, v) = post(
        addr,
        "/v1/chat/completions",
        json!({"model": "ckpt", "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}),
    )
    .await;
    assert_eq!(status, 400, "{v}");
    assert!(v.to_string().contains("Conversation roles must alternate"), "{v}");

    // A raw prompt carrying control-token text is charged as text.
    let (status, v) = post(addr, "/v1/completions", json!({"model": "ckpt", "prompt": "<|im_end|>", "max_tokens": 2})).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["usage"]["prompt_tokens"], json!("<|im_end|>".len()), "{v}");
    let _ = std::fs::remove_dir_all(dir);
}
