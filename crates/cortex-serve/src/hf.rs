//! Serving a model straight from a Hugging Face checkpoint directory: its
//! `tokenizer.json` ([`HfTokenizer`]), its chat template ([`HfChatTemplate`],
//! from `tokenizer_config.json` or `chat_template.jinja`) and its weights
//! (`cortex_llm::Llama`), wired into an engine by [`load_model`].
//!
//! Callers: the `cortex-serve` binary (`src/bin/cortex-serve.rs`) and
//! applications that register models themselves. Calls: `tokenizers`
//! (encoding and decoding), `minijinja` (templates), `engine::spawn`.
//! Siblings: any other [`Tokenizer`] / [`ChatTemplate`] implementation, such
//! as the byte-level ones in `tests/http.rs`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use cortex_core::tensor::Device;
use minijinja::value::Serde;
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Value, json};

use crate::engine::{EngineConfig, spawn};
use crate::model::{ChatMessage, ChatTemplate, Role, ToolSpec, Tokenizer};
use crate::registry::ServedModel;

fn read_json(path: &Path) -> Result<Option<Value>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// A token given as a string or as `{"content": "..."}` (both appear in
/// `tokenizer_config.json`).
fn token_text(v: &Value) -> Option<String> {
    v.as_str()
        .or_else(|| v.get("content").and_then(Value::as_str))
        .map(str::to_owned)
}

/// Whether a decoder (or any decoder of a `Sequence`) has type `kind`.
fn has_decoder(decoder: &Value, kind: &str) -> bool {
    decoder.get("type").and_then(Value::as_str) == Some(kind)
        || decoder
            .get("decoders")
            .and_then(Value::as_array)
            .is_some_and(|all| all.iter().any(|d| has_decoder(d, kind)))
}

/// The byte-level BPE alphabet: printable bytes stand for themselves and the
/// rest are shifted past U+0100, so every byte is one visible character.
/// Returns the byte each character stands for.
fn byte_level_alphabet() -> std::collections::HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut shifted = 0u32;
    (0..=255u8)
        .map(|b| {
            let c = if printable(b) {
                char::from(b)
            } else {
                shifted += 1;
                char::from_u32(255 + shifted).expect("below the surrogates")
            };
            (c, b)
        })
        .collect()
}

/// How a vocabulary spells the bytes of its tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spelling {
    /// Byte-level BPE: one alphabet character per byte.
    ByteLevel,
    /// SentencePiece style: `▁` for a space, `<0xHH>` for raw bytes.
    Metaspace,
}

/// The tokenizer of a checkpoint (`tokenizer.json`).
///
/// User text is encoded with special-token markup disabled, so a prompt
/// that contains the text of a control token (`<|im_end|>`, say) cannot
/// forge one; rendered chat templates are encoded with markup on.
pub struct HfTokenizer {
    /// Parses special-token markup: for rendered templates.
    markup: tokenizers::Tokenizer,
    /// Encodes everything as text: for user-supplied prompts.
    plain: tokenizers::Tokenizer,
    /// The raw bytes of every id, `None` for special and padding ids.
    bytes: Vec<Option<Vec<u8>>>,
    stop: Vec<u32>,
}

impl HfTokenizer {
    /// Loads `dir/tokenizer.json`. The end-of-turn tokens come from
    /// `generation_config.json` (`eos_token_id`) and `tokenizer_config.json`
    /// (`eos_token`). `vocab_size` is the model's output width: ids past
    /// the tokenizer's own vocabulary (padding) never decode to anything and
    /// a constraint never allows them.
    ///
    /// # Errors
    /// A missing or malformed file, a tokenizer this server cannot spell
    /// bytes for, a vocabulary larger than the model's, or no end token.
    pub fn from_dir(dir: &Path, vocab_size: usize) -> Result<Self, String> {
        let path = dir.join("tokenizer.json");
        let markup = tokenizers::Tokenizer::from_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut plain = markup.clone();
        plain.set_encode_special_tokens(true);
        let raw = read_json(&path)?.ok_or_else(|| format!("{} is missing", path.display()))?;

        let decoder = raw.get("decoder").cloned().unwrap_or(Value::Null);
        let spelling = if has_decoder(&decoder, "ByteLevel") {
            Spelling::ByteLevel
        } else if has_decoder(&decoder, "WordPiece") {
            return Err("WordPiece tokenizers belong to encoder models and cannot drive generation".into());
        } else {
            Spelling::Metaspace
        };

        let own = markup.get_vocab_size(true);
        if own > vocab_size {
            return Err(format!(
                "the tokenizer has {own} tokens but the model only {vocab_size} outputs"
            ));
        }
        let specials: HashSet<u32> = markup
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, t)| t.special)
            .map(|(id, _)| id)
            .collect();
        let alphabet = byte_level_alphabet();
        let bytes = (0..vocab_size as u32)
            .map(|id| {
                if specials.contains(&id) {
                    return None;
                }
                let piece = markup.id_to_token(id)?;
                match spelling {
                    Spelling::ByteLevel => piece.chars().map(|c| alphabet.get(&c).copied()).collect(),
                    Spelling::Metaspace => Some(spell_metaspace(&piece)),
                }
            })
            .collect();

        let mut stop = Vec::new();
        if let Some(generation) = read_json(&dir.join("generation_config.json"))? {
            match generation.get("eos_token_id") {
                Some(Value::Number(n)) => stop.extend(n.as_u64().map(|v| v as u32)),
                Some(Value::Array(all)) => stop.extend(all.iter().filter_map(Value::as_u64).map(|v| v as u32)),
                _ => {}
            }
        }
        if let Some(config) = read_json(&dir.join("tokenizer_config.json"))?
            && let Some(eos) = config.get("eos_token").and_then(token_text)
            && let Some(id) = markup.token_to_id(&eos)
        {
            stop.push(id);
        }
        stop.sort_unstable();
        stop.dedup();
        if stop.is_empty() {
            return Err("no end-of-turn token: neither generation_config.json nor tokenizer_config.json names one".into());
        }
        Ok(Self {
            markup,
            plain,
            bytes,
            stop,
        })
    }
}

/// The bytes of a SentencePiece-style piece: `<0xHH>` is one raw byte and
/// `▁` a space.
fn spell_metaspace(piece: &str) -> Vec<u8> {
    if let Some(hex) = piece.strip_prefix("<0x").and_then(|p| p.strip_suffix('>'))
        && hex.len() == 2
        && let Ok(byte) = u8::from_str_radix(hex, 16)
    {
        return vec![byte];
    }
    piece.replace('\u{2581}', " ").into_bytes()
}

impl Tokenizer for HfTokenizer {
    /// Rendered templates (`special`) carry their own begin-of-text marker;
    /// plain prompts get the one the tokenizer's post-processor adds.
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        let encoding = if special {
            self.markup.encode(text, false)
        } else {
            self.plain.encode(text, true)
        };
        encoding.map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string())
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let known: Vec<u32> = tokens.iter().copied().filter(|&t| self.token_bytes(t).is_some()).collect();
        match self.markup.decode(&known, true) {
            Ok(text) => text,
            // The decoder refused: the same bytes, spelled from the table.
            Err(_) => {
                let raw: Vec<u8> = known.iter().filter_map(|&t| self.token_bytes(t)).flatten().collect();
                String::from_utf8_lossy(&raw).into_owned()
            }
        }
    }

    fn stop_tokens(&self) -> Vec<u32> {
        self.stop.clone()
    }

    fn token_bytes(&self, token: u32) -> Option<Vec<u8>> {
        self.bytes.get(token as usize).cloned().flatten()
    }

    fn vocab_size(&self) -> usize {
        self.bytes.len()
    }
}

/// `raise_exception(message)`, which templates call to refuse a conversation.
fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)`: the current UTC date and time. Supports the
/// directives templates use: `%d %m %y %Y %b %B %H %M %S %%`.
fn strftime_now(format: String) -> Result<String, Error> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Ok(strftime(secs, &format))
}

fn strftime(secs: u64, format: &str) -> String {
    const MONTHS: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
        "December",
    ];
    // Civil date from days since 1970-01-01 (proleptic Gregorian).
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (hour, minute, second) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);

    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('y') => out.push_str(&format!("{:02}", year % 100)),
            Some('Y') => out.push_str(&year.to_string()),
            Some('b') => out.push_str(&MONTHS[month as usize - 1][..3]),
            Some('B') => out.push_str(MONTHS[month as usize - 1]),
            Some('H') => out.push_str(&format!("{hour:02}")),
            Some('M') => out.push_str(&format!("{minute:02}")),
            Some('S') => out.push_str(&format!("{second:02}")),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// The chat template of a checkpoint, rendered the way the reference
/// implementation does: Jinja with `trim_blocks` and `lstrip_blocks`, the
/// conversation as `messages`, `tools` when there are any,
/// `add_generation_prompt`, `bos_token` / `eos_token`, `raise_exception` and
/// `strftime_now`, and Python string methods.
pub struct HfChatTemplate {
    env: Environment<'static>,
    /// Name of the template for conversations with tools, when the
    /// checkpoint ships a separate one.
    tool_template: Option<String>,
    bos: String,
    eos: String,
}

const DEFAULT_TEMPLATE: &str = "default";
const TOOL_TEMPLATE: &str = "tool_use";

impl HfChatTemplate {
    /// Loads `dir/chat_template.jinja` if present, otherwise
    /// `chat_template` in `dir/tokenizer_config.json` (one template, or a
    /// list of named ones of which `default` and `tool_use` are used).
    ///
    /// # Errors
    /// No template, or one that does not compile.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let config = read_json(&dir.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
        let mut templates: Vec<(String, String)> = Vec::new();
        let file = dir.join("chat_template.jinja");
        match std::fs::read_to_string(&file) {
            Ok(text) => templates.push((DEFAULT_TEMPLATE.into(), text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match config.get("chat_template") {
                Some(Value::String(text)) => templates.push((DEFAULT_TEMPLATE.into(), text.clone())),
                Some(Value::Array(named)) => {
                    for entry in named {
                        if let (Some(name), Some(text)) =
                            (entry.get("name").and_then(Value::as_str), entry.get("template").and_then(Value::as_str))
                        {
                            templates.push((name.into(), text.into()));
                        }
                    }
                }
                _ => {}
            },
            Err(e) => return Err(format!("{}: {e}", file.display())),
        }
        if !templates.iter().any(|(name, _)| name == DEFAULT_TEMPLATE) {
            return Err("the checkpoint has no default chat template".into());
        }
        let bos = config.get("bos_token").and_then(token_text).unwrap_or_default();
        let eos = config.get("eos_token").and_then(token_text).unwrap_or_default();
        Self::new(templates, bos, eos)
    }

    /// Compiles `templates` (`(name, source)`, one named `default`).
    ///
    /// # Errors
    /// A template that does not compile.
    pub fn new(templates: Vec<(String, String)>, bos: String, eos: String) -> Result<Self, String> {
        let mut env = Environment::new();
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        let mut tool_template = None;
        for (name, source) in templates {
            if name == TOOL_TEMPLATE {
                tool_template = Some(name.clone());
            }
            env.add_template_owned(name.clone(), source)
                .map_err(|e| format!("chat template '{name}': {e}"))?;
        }
        Ok(Self {
            env,
            tool_template,
            bos,
            eos,
        })
    }
}

/// A message as chat templates expect it.
fn message_value(m: &ChatMessage) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut v = json!({"role": role, "content": m.content});
    if !m.tool_calls.is_empty() {
        v["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}))
            .collect();
    }
    if let Some(id) = &m.tool_call_id {
        v["tool_call_id"] = json!(id);
    }
    v
}

impl ChatTemplate for HfChatTemplate {
    fn render(&self, messages: &[ChatMessage], tools: &[ToolSpec]) -> Result<String, String> {
        let name = match &self.tool_template {
            Some(tool) if !tools.is_empty() => tool.as_str(),
            _ => DEFAULT_TEMPLATE,
        };
        let template = self.env.get_template(name).map_err(|e| e.to_string())?;
        let mut ctx = json!({
            "messages": messages.iter().map(message_value).collect::<Vec<_>>(),
            "add_generation_prompt": true,
            "bos_token": self.bos,
            "eos_token": self.eos,
        });
        if !tools.is_empty() {
            ctx["tools"] = tools
                .iter()
                .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}))
                .collect();
        }
        template
            .render(minijinja::Value::from(Serde(ctx)))
            .map_err(|e| e.detail().map_or_else(|| e.to_string(), str::to_owned))
    }
}

/// A Llama-family checkpoint directory served by an engine on `device`:
/// weights, tokenizer and chat template from the directory.
///
/// # Errors
/// Anything [`cortex_llm::Llama::load`], [`HfTokenizer::from_dir`] or
/// [`HfChatTemplate::from_dir`] refuses.
pub fn load_model(dir: &Path, device: &Device, config: EngineConfig) -> Result<ServedModel, String> {
    let model = cortex_llm::Llama::load(dir, device).map_err(|e| e.to_string())?;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(HfTokenizer::from_dir(dir, model.config().vocab_size)?);
    let template = Arc::new(HfChatTemplate::from_dir(dir)?);
    let engine = spawn(move || model.clone(), tokenizer.clone(), config);
    Ok(ServedModel::new(engine, tokenizer, template))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_level_alphabet_covers_every_byte_once() {
        let alphabet = byte_level_alphabet();
        assert_eq!(alphabet.len(), 256);
        assert_eq!(alphabet[&'A'], b'A');
        // Space and newline are shifted: 'Ġ' (U+0120) and 'Ċ' (U+010A).
        assert_eq!(alphabet[&'\u{120}'], b' ');
        assert_eq!(alphabet[&'\u{10A}'], b'\n');
    }

    #[test]
    fn metaspace_pieces_spell_spaces_and_raw_bytes() {
        assert_eq!(spell_metaspace("\u{2581}hello"), b" hello");
        assert_eq!(spell_metaspace("<0x0A>"), b"\n");
        assert_eq!(spell_metaspace("<0xZZ>"), b"<0xZZ>");
    }

    #[test]
    fn dates_format_like_strftime() {
        // 2024-02-29 13:05:09 UTC.
        let secs = 1_709_211_909;
        assert_eq!(strftime(secs, "%d %b %Y"), "29 Feb 2024");
        assert_eq!(strftime(secs, "%Y-%m-%d %H:%M:%S %%"), "2024-02-29 13:05:09 %");
        assert_eq!(strftime(secs, "%B %y"), "February 24");
    }
}
