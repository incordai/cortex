//! Tool-call extraction from model output, as it streams.
//!
//! Recognized forms, which cover the common chat templates:
//! - `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks
//!   (Hermes / Qwen), anywhere in the text, any number of them;
//! - an output that starts with a call object `{"name": ..., "arguments" |
//!   "parameters": {...}}`, or a list of them (Llama 3 style), whose name is
//!   one of the offered tools (so an ordinary JSON answer is not a call).
//!
//! [`ToolParser`] reads the output incrementally: a call is reported as
//! soon as its name is known ([`ToolEvent::CallStart`]), its arguments as
//! they are written ([`ToolEvent::CallArgs`]), and its parsed form when it
//! closes ([`ToolEvent::CallEnd`]). Text that cannot be part of a call is
//! released immediately; text that might be is held only until it is
//! decided. Anything that turns out not to be a call is released as text,
//! so a malformed call is shown to the client rather than dropped.
//! [`parse_tool_calls`] runs the same parser over a whole output, so the
//! streamed and collected forms always agree.
//!
//! Callers: the OpenAI, Anthropic and Ollama chat handlers in `api/`.

use serde_json::Value;

use crate::model::ToolCall;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// What the parser found in the latest piece of output.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolEvent {
    /// Plain text.
    Text(String),
    /// Call number `index` began.
    CallStart {
        /// Position among this output's calls.
        index: usize,
        /// Fresh call identifier.
        id: String,
        /// The function called.
        name: String,
    },
    /// More of call `index`'s arguments, as JSON text.
    CallArgs {
        /// The call.
        index: usize,
        /// The next piece of the arguments' JSON.
        delta: String,
    },
    /// Call `index` is complete. `call` is `None` when its text never became
    /// a valid call (e.g. the output was cut off); `raw` is that text.
    CallEnd {
        /// The call.
        index: usize,
        /// The parsed call.
        call: Option<ToolCall>,
        /// The call's text as written.
        raw: String,
    },
}

/// The name and arguments of a call object, if it is one.
fn call_parts(value: &Value) -> Option<(String, Value)> {
    let obj = value.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let arguments = match obj.get("arguments").or_else(|| obj.get("parameters")) {
        // Some models emit the arguments as a JSON string.
        Some(Value::String(s)) => serde_json::from_str(s).ok()?,
        Some(v @ Value::Object(_)) => v.clone(),
        None => Value::Object(Default::default()),
        Some(_) => return None,
    };
    Some((name, arguments))
}

fn is_args_key(key: &str) -> bool {
    key == "arguments" || key == "parameters"
}

/// Where the scan of a call object's top level stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Before the opening `{`.
    Before,
    /// A key or the end.
    Key,
    InKey,
    Colon,
    Value,
    /// Inside a top-level value of this kind.
    InValue(Kind),
    After,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Str,
    Container,
    Scalar,
}

/// One call object being read: `{"name": ..., "arguments": ...}`.
#[derive(Debug)]
struct CallScan {
    text: String,
    pos: usize,
    depth: u32,
    in_str: bool,
    escape: bool,
    phase: Phase,
    key_start: usize,
    key: String,
    value_start: usize,
    first_key: Option<String>,
    name: Option<String>,
    /// Start and (once known) end of the arguments value, and whether it is
    /// a JSON string (decoded only at the end).
    args: Option<(usize, Option<usize>, bool)>,
    sent: usize,
    /// Index and id, once the call has been reported.
    started: Option<(usize, String)>,
    /// End offset of the object once it closed.
    closed: Option<usize>,
    /// Not a JSON object after all.
    broken: bool,
}

impl CallScan {
    fn new() -> Self {
        Self {
            text: String::new(),
            pos: 0,
            depth: 0,
            in_str: false,
            escape: false,
            phase: Phase::Before,
            key_start: 0,
            key: String::new(),
            value_start: 0,
            first_key: None,
            name: None,
            args: None,
            sent: 0,
            started: None,
            closed: None,
            broken: false,
        }
    }

    /// A value at the top level ended at `end`.
    fn value_end(&mut self, end: usize) {
        let raw = &self.text[self.value_start..end];
        if self.key == "name" {
            self.name = serde_json::from_str::<String>(raw).ok();
        } else if is_args_key(&self.key) && self.args.is_none() {
            let string = raw.starts_with('"');
            self.args = Some((self.value_start, Some(end), string));
        }
        self.phase = Phase::After;
    }

    /// Scans the text received so far, stopping where the object closes.
    fn scan(&mut self) {
        while self.pos < self.text.len() && self.closed.is_none() && !self.broken {
            let i = self.pos;
            let c = self.text.as_bytes()[i];
            self.pos += 1;
            if self.in_str {
                if self.escape {
                    self.escape = false;
                } else if c == b'\\' {
                    self.escape = true;
                } else if c == b'"' {
                    self.in_str = false;
                    if self.depth == 1 {
                        match self.phase {
                            Phase::InKey => {
                                // Keys compare unescaped: `"arguments"` is `arguments`.
                                let raw = &self.text[self.key_start..=i];
                                self.key = serde_json::from_str::<String>(raw).unwrap_or_else(|_| raw.to_string());
                                if self.first_key.is_none() {
                                    self.first_key = Some(self.key.clone());
                                }
                                self.phase = Phase::Colon;
                            }
                            Phase::InValue(Kind::Str) => self.value_end(i + 1),
                            _ => {}
                        }
                    }
                }
                continue;
            }
            // A scalar value at the top level ends at the first byte that
            // cannot continue it.
            if self.depth == 1
                && self.phase == Phase::InValue(Kind::Scalar)
                && matches!(c, b',' | b'}' | b' ' | b'\t' | b'\n' | b'\r')
            {
                self.value_end(i);
            }
            match c {
                b' ' | b'\t' | b'\n' | b'\r' => {}
                b'"' => {
                    self.in_str = true;
                    if self.depth == 0 {
                        self.broken = true;
                    } else if self.depth == 1 {
                        match self.phase {
                            Phase::Key => {
                                self.phase = Phase::InKey;
                                self.key_start = i;
                            }
                            Phase::Value => {
                                self.phase = Phase::InValue(Kind::Str);
                                self.value_start = i;
                            }
                            Phase::InValue(_) => {}
                            _ => self.broken = true,
                        }
                    }
                }
                b'{' | b'[' => {
                    if self.depth == 0 {
                        if c != b'{' || self.phase != Phase::Before {
                            self.broken = true;
                        }
                        self.phase = Phase::Key;
                    } else if self.depth == 1 {
                        if self.phase == Phase::Value {
                            self.phase = Phase::InValue(Kind::Container);
                            self.value_start = i;
                            if is_args_key(&self.key) && self.args.is_none() {
                                self.args = Some((i, None, false));
                            }
                        } else {
                            self.broken = true;
                        }
                    }
                    self.depth += 1;
                }
                b'}' | b']' => {
                    if self.depth == 0 {
                        self.broken = true;
                        continue;
                    }
                    self.depth -= 1;
                    if self.depth == 1 && self.phase == Phase::InValue(Kind::Container) {
                        if is_args_key(&self.key)
                            && let Some((start, None, false)) = self.args
                        {
                            self.args = Some((start, Some(i + 1), false));
                        }
                        self.value_end(i + 1);
                    } else if self.depth == 0 {
                        if c != b'}' || !matches!(self.phase, Phase::Key | Phase::After) {
                            self.broken = true;
                        }
                        self.closed = Some(i + 1);
                    }
                }
                b':' if self.depth == 1 => {
                    if self.phase == Phase::Colon {
                        self.phase = Phase::Value;
                    } else {
                        self.broken = true;
                    }
                }
                b',' if self.depth == 1 => {
                    if self.phase == Phase::After {
                        self.phase = Phase::Key;
                    } else {
                        self.broken = true;
                    }
                }
                _ if self.depth == 0 => self.broken = true,
                _ if self.depth == 1 => match self.phase {
                    Phase::Value => {
                        self.phase = Phase::InValue(Kind::Scalar);
                        self.value_start = i;
                    }
                    Phase::InValue(Kind::Scalar) => {}
                    _ => self.broken = true,
                },
                _ => {}
            }
        }
    }

    /// Reports the call once its name is known, then any new argument text.
    fn progress(&mut self, ctx: &mut Ctx<'_>, require_known: bool, out: &mut Vec<ToolEvent>) {
        if self.started.is_none()
            && let Some(name) = &self.name
            && (!require_known || ctx.names.iter().any(|n| n == name))
        {
            let index = ctx.next_index;
            ctx.next_index += 1;
            let id = (ctx.ids)();
            out.push(ToolEvent::CallStart {
                index,
                id: id.clone(),
                name: name.clone(),
            });
            self.started = Some((index, id));
        }
        let Some((index, _)) = &self.started else {
            return;
        };
        if let Some((start, end, false)) = self.args {
            let end = end.unwrap_or(self.pos);
            let from = self.sent.max(start);
            if end > from {
                out.push(ToolEvent::CallArgs {
                    index: *index,
                    delta: self.text[from..end].to_string(),
                });
                self.sent = end;
            }
        }
    }

    /// Ends a reported call; `false` (and nothing emitted) if it was never
    /// reported. `raw` is the call's text as written.
    fn end(&mut self, raw: &str, out: &mut Vec<ToolEvent>) -> bool {
        let Some((index, id)) = self.started.take() else {
            return false;
        };
        let parsed = self
            .closed
            .filter(|_| !self.broken)
            .and_then(|end| serde_json::from_str::<Value>(&self.text[..end]).ok())
            .and_then(|v| call_parts(&v));
        // A string-valued argument object arrives whole, decoded.
        if let (Some((_, args)), Some((_, _, true))) = (&parsed, self.args) {
            out.push(ToolEvent::CallArgs {
                index,
                delta: args.to_string(),
            });
        }
        out.push(ToolEvent::CallEnd {
            index,
            call: parsed.map(|(name, arguments)| ToolCall { id, name, arguments }),
            raw: raw.to_string(),
        });
        true
    }

    /// Whether, in bare mode, this object can no longer be a call (it is
    /// broken, starts with another key, or names an unknown tool).
    fn not_a_call(&self, names: &[String]) -> bool {
        self.broken
            || self.first_key.as_deref().is_some_and(|k| k != "name" && !is_args_key(k))
            || self.name.as_ref().is_some_and(|n| !names.contains(n))
    }
}

/// What every scan needs from the parser.
struct Ctx<'a> {
    names: &'a [String],
    ids: &'a mut (dyn FnMut() -> String + Send),
    next_index: usize,
}

/// The longest suffix of `text` that is a proper prefix of `tag`.
fn partial_tag(text: &str, tag: &str) -> usize {
    (1..tag.len().min(text.len() + 1))
        .rev()
        .find(|&n| text.is_char_boundary(text.len() - n) && tag.starts_with(&text[text.len() - n..]))
        .unwrap_or(0)
}

#[derive(Debug)]
enum Mode {
    /// No visible output yet; holds leading whitespace.
    Start(String),
    /// Plain text; holds a possible partial `<tool_call>`.
    Text(String),
    /// Inside a `<tool_call>` block.
    Tagged { scan: CallScan, held: String },
    /// The output began with `{` or `[`: a bare call or list of calls.
    Bare(Bare),
}

#[derive(Debug)]
struct Bare {
    /// Everything of this region, released as text if it is not a call.
    raw: String,
    list: bool,
    /// The call object being read.
    scan: Option<CallScan>,
    /// In a list: an element may start (after `[` or `,`).
    expect_item: bool,
    /// Whether any call of the region has been reported.
    reported: bool,
}

/// Splits model output into text and tool calls as it streams.
pub struct ToolParser {
    names: Vec<String>,
    ids: Box<dyn FnMut() -> String + Send>,
    next_index: usize,
    mode: Mode,
}

impl ToolParser {
    /// A parser for an output that may call the tools in `names`; `ids`
    /// makes fresh call identifiers.
    pub fn new(names: Vec<String>, ids: impl FnMut() -> String + Send + 'static) -> Self {
        Self {
            names,
            ids: Box::new(ids),
            next_index: 0,
            mode: Mode::Start(String::new()),
        }
    }

    /// Whether any call has been reported.
    pub fn has_calls(&self) -> bool {
        self.next_index > 0
    }

    /// Reads the next piece of output.
    pub fn push(&mut self, text: &str) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        self.feed(text.to_string(), &mut out);
        out
    }

    /// Ends the output, releasing whatever is held.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        let mut ctx = Ctx {
            names: &self.names,
            ids: &mut *self.ids,
            next_index: self.next_index,
        };
        match std::mem::replace(&mut self.mode, Mode::Text(String::new())) {
            Mode::Start(held) | Mode::Text(held) => push_text(&mut out, held),
            Mode::Tagged { mut scan, held } => {
                scan.text.push_str(&held);
                scan.scan();
                scan.progress(&mut ctx, false, &mut out);
                let raw = format!("{OPEN}{}", scan.text);
                if !scan.end(&raw, &mut out) {
                    push_text(&mut out, raw);
                }
            }
            Mode::Bare(mut bare) => match bare.scan.take() {
                // The output ended inside a reported call.
                Some(mut scan) if scan.started.is_some() => {
                    let raw = scan.text.clone();
                    scan.end(&raw, &mut out);
                }
                // An element that never became a call, after others did.
                Some(scan) if bare.reported => push_text(&mut out, scan.text),
                _ if !bare.reported => push_text(&mut out, bare.raw),
                _ => {}
            },
        }
        self.next_index = ctx.next_index;
        out
    }

    fn feed(&mut self, text: String, out: &mut Vec<ToolEvent>) {
        let mut pending = text;
        loop {
            let mut ctx = Ctx {
                names: &self.names,
                ids: &mut *self.ids,
                next_index: self.next_index,
            };
            let (mode, rest) = step(std::mem::replace(&mut self.mode, Mode::Text(String::new())), pending, &mut ctx, out);
            self.next_index = ctx.next_index;
            self.mode = mode;
            match rest {
                Some(r) if !r.is_empty() => pending = r,
                _ => return,
            }
        }
    }
}

fn push_text(out: &mut Vec<ToolEvent>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(ToolEvent::Text(last)) = out.last_mut() {
        last.push_str(&text);
    } else {
        out.push(ToolEvent::Text(text));
    }
}

/// Advances `mode` over `input`. Returns the next mode and the input it did
/// not consume (handed to that mode).
fn step(mode: Mode, input: String, ctx: &mut Ctx<'_>, out: &mut Vec<ToolEvent>) -> (Mode, Option<String>) {
    match mode {
        Mode::Start(mut held) => {
            held.push_str(&input);
            let Some(first) = held.find(|c: char| !c.is_whitespace()) else {
                return (Mode::Start(held), None);
            };
            match held.as_bytes()[first] {
                b @ (b'{' | b'[') => {
                    let lead = held[..first].to_string();
                    let body = held[first..].to_string();
                    let bare = Bare {
                        raw: lead,
                        list: b == b'[',
                        scan: None,
                        expect_item: b == b'[',
                        reported: false,
                    };
                    (Mode::Bare(bare), Some(body))
                }
                _ => (Mode::Text(String::new()), Some(held)),
            }
        }
        Mode::Text(mut held) => {
            held.push_str(&input);
            if let Some(at) = held.find(OPEN) {
                push_text(out, held[..at].to_string());
                let rest = held[at + OPEN.len()..].to_string();
                return (
                    Mode::Tagged {
                        scan: CallScan::new(),
                        held: String::new(),
                    },
                    Some(rest),
                );
            }
            let keep = partial_tag(&held, OPEN);
            let emit = held[..held.len() - keep].to_string();
            push_text(out, emit);
            (Mode::Text(held[held.len() - keep..].to_string()), None)
        }
        Mode::Tagged { mut scan, mut held } => {
            held.push_str(&input);
            if let Some(at) = held.find(CLOSE) {
                scan.text.push_str(&held[..at]);
                scan.scan();
                scan.progress(ctx, false, out);
                let raw = format!("{OPEN}{}{CLOSE}", scan.text);
                if !scan.end(&raw, out) {
                    push_text(out, raw);
                }
                let rest = held[at + CLOSE.len()..].to_string();
                return (Mode::Text(String::new()), Some(rest));
            }
            let keep = partial_tag(&held, CLOSE);
            scan.text.push_str(&held[..held.len() - keep]);
            scan.scan();
            scan.progress(ctx, false, out);
            (
                Mode::Tagged {
                    scan,
                    held: held[held.len() - keep..].to_string(),
                },
                None,
            )
        }
        Mode::Bare(bare) => bare_step(bare, input, ctx, out),
    }
}

fn bare_step(mut bare: Bare, input: String, ctx: &mut Ctx<'_>, out: &mut Vec<ToolEvent>) -> (Mode, Option<String>) {
    let mut input = input;
    loop {
        if let Some(mut scan) = bare.scan.take() {
            bare.raw.push_str(&input);
            scan.text.push_str(&input);
            scan.scan();
            let bad = scan.not_a_call(ctx.names);
            if !bad {
                scan.progress(ctx, true, out);
                bare.reported |= scan.started.is_some();
            }
            if bad || (scan.closed.is_some() && scan.started.is_none()) {
                if scan.started.is_some() {
                    // A reported call broke off: end it; the rest is text.
                    let raw = scan.text.clone();
                    scan.end(&raw, out);
                    return (Mode::Text(String::new()), None);
                }
                if !bare.reported {
                    // Not a call after all: everything so far is text.
                    let raw = std::mem::take(&mut bare.raw);
                    return (Mode::Text(String::new()), Some(raw));
                }
                // A later list element that is not a call: the rest of the
                // output passes through as text.
                return (Mode::Text(String::new()), Some(scan.text));
            }
            let Some(end) = scan.closed else {
                bare.scan = Some(scan);
                return (Mode::Bare(bare), None);
            };
            let leftover = scan.text[end..].to_string();
            let body = scan.text[..end].to_string();
            scan.end(&body, out);
            if !bare.list {
                return (Mode::Text(String::new()), Some(leftover));
            }
            input = leftover;
            continue;
        }
        // Between elements (or before the single object).
        let Some(first) = input.find(|c: char| !c.is_whitespace()) else {
            bare.raw.push_str(&input);
            return (Mode::Bare(bare), None);
        };
        let c = input.as_bytes()[first];
        match c {
            b'[' if bare.list && bare.raw.trim().is_empty() && !bare.reported => {
                bare.raw.push_str(&input[..=first]);
                input = input[first + 1..].to_string();
                bare.expect_item = true;
            }
            b'{' if !bare.list || bare.expect_item => {
                bare.raw.push_str(&input[..first]);
                bare.expect_item = false;
                bare.scan = Some(CallScan::new());
                input = input[first..].to_string();
            }
            b',' if bare.list && !bare.expect_item && bare.reported => {
                bare.raw.push_str(&input[..=first]);
                bare.expect_item = true;
                input = input[first + 1..].to_string();
            }
            b']' if bare.list && bare.reported && !bare.expect_item => {
                // The list is complete; what follows is text.
                return (Mode::Text(String::new()), Some(input[first + 1..].to_string()));
            }
            _ => {
                if bare.reported {
                    return (Mode::Text(String::new()), Some(input[first..].to_string()));
                }
                bare.raw.push_str(&input);
                let raw = std::mem::take(&mut bare.raw);
                return (Mode::Text(String::new()), Some(raw));
            }
        }
    }
}

/// Splits a whole output into its text and the tool calls it holds, exactly
/// as [`ToolParser`] would while streaming it. `names` are the offered
/// tools; `id` makes fresh call identifiers.
pub fn parse_tool_calls(text: &str, names: &[String], id: impl FnMut() -> String + Send + 'static) -> (String, Vec<ToolCall>) {
    let mut parser = ToolParser::new(names.to_vec(), id);
    let mut events = parser.push(text);
    events.extend(parser.finish());
    let mut rest = String::new();
    let mut calls = Vec::new();
    for event in events {
        match event {
            ToolEvent::Text(t) => rest.push_str(&t),
            ToolEvent::CallEnd { call: Some(call), .. } => calls.push(call),
            ToolEvent::CallEnd { call: None, raw, .. } => rest.push_str(&raw),
            ToolEvent::CallStart { .. } | ToolEvent::CallArgs { .. } => {}
        }
    }
    if calls.is_empty() {
        (rest, calls)
    } else {
        (rest.trim().to_string(), calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ids() -> impl FnMut() -> String + Send + 'static {
        let mut n = 0;
        move || {
            n += 1;
            format!("call_{n}")
        }
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn tagged_blocks_are_extracted_and_the_rest_kept() {
        let text = "Let me check.<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>\n<tool_call>{\"name\":\"time\",\"arguments\":\"{\\\"tz\\\":\\\"UTC\\\"}\"}</tool_call>";
        let (rest, calls) = parse_tool_calls(text, &names(&["weather", "time"]), ids());
        assert_eq!(rest, "Let me check.");
        assert_eq!(calls.len(), 2);
        assert_eq!((calls[0].id.as_str(), calls[0].name.as_str()), ("call_1", "weather"));
        assert_eq!(calls[0].arguments, json!({"city": "Oslo"}));
        assert_eq!(calls[1].arguments, json!({"tz": "UTC"}), "string arguments are decoded");
    }

    #[test]
    fn malformed_blocks_stay_in_the_text() {
        let text = "<tool_call>{not json}</tool_call> done";
        let (rest, calls) = parse_tool_calls(text, &names(&["f"]), ids());
        assert!(calls.is_empty());
        assert_eq!(rest, text);
    }

    #[test]
    fn a_bare_call_object_or_list_is_a_call() {
        let (rest, calls) = parse_tool_calls(r#" {"name": "f", "parameters": {"x": 1}} "#, &names(&["f"]), ids());
        assert_eq!((rest.as_str(), calls.len()), ("", 1));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        let (_, calls) = parse_tool_calls(r#"[{"name": "a"}, {"name": "b", "arguments": {}}]"#, &names(&["a", "b"]), ids());
        assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn ordinary_json_answers_are_not_calls() {
        for text in [r#"{"answer": 42}"#, r#"{"name": "Bob", "age": 3}"#, "[1, 2]", r#"[{"x": 1}]"#] {
            let (rest, calls) = parse_tool_calls(text, &names(&["lookup"]), ids());
            assert!(calls.is_empty(), "{text}");
            assert_eq!(rest, text);
        }
    }

    /// Feeds `text` in pieces of `size` bytes (at char boundaries).
    fn stream(text: &str, size: usize, tools: &[&str]) -> Vec<ToolEvent> {
        let mut p = ToolParser::new(names(tools), ids());
        let mut events = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let mut end = (at + size).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            events.extend(p.push(&text[at..end]));
            at = end;
        }
        events.extend(p.finish());
        events
    }

    #[test]
    fn calls_stream_before_the_output_ends() {
        let text = "Sure.<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": [1, 2]}}</tool_call> Done.";
        let mut p = ToolParser::new(names(&["weather"]), ids());
        let mut seen_start_at = None;
        let mut args = String::new();
        for (i, ch) in text.char_indices() {
            for e in p.push(&ch.to_string()) {
                match e {
                    ToolEvent::CallStart { name, .. } => {
                        assert_eq!(name, "weather");
                        seen_start_at = Some(i);
                    }
                    ToolEvent::CallArgs { delta, .. } => args.push_str(&delta),
                    _ => {}
                }
            }
        }
        p.finish();
        let started = seen_start_at.expect("the call was reported");
        assert!(started < text.find("Oslo").unwrap(), "reported as soon as the name was known");
        assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"city": "Oslo", "days": [1, 2]}));
    }

    #[test]
    fn streamed_and_whole_parses_agree_at_any_split() {
        let cases = [
            ("Hi <tool_call>{\"name\":\"a\",\"arguments\":{\"q\":\"x}\"}}</tool_call> bye <tool_call>{\"name\":\"b\"}</tool_call>", vec!["a", "b"]),
            ("[{\"name\": \"a\", \"arguments\": {\"n\": 1}}, {\"name\": \"b\", \"parameters\": {}}] tail", vec!["a", "b"]),
            ("{\"arguments\": {\"k\": [1]}, \"name\": \"a\"}", vec!["a"]),
            ("{\"name\": \"zzz\", \"arguments\": {}}", vec!["a"]),
            ("plain <tool_ text with < signs", vec!["a"]),
            ("<tool_call>{\"name\": \"a\", \"arguments\": {\"cut", vec!["a"]),
        ];
        for (text, tools) in cases {
            let tools: Vec<String> = tools.iter().map(|s| s.to_string()).collect();
            let whole = parse_tool_calls(text, &tools, ids());
            for size in [1, 2, 3, 7, 1000] {
                let events = stream(text, size, &tools.iter().map(String::as_str).collect::<Vec<_>>());
                let mut rest = String::new();
                let mut calls = Vec::new();
                let mut args: Vec<String> = Vec::new();
                for e in events {
                    match e {
                        ToolEvent::Text(t) => rest.push_str(&t),
                        ToolEvent::CallStart { index, .. } => {
                            assert_eq!(index, args.len());
                            args.push(String::new());
                        }
                        ToolEvent::CallArgs { index, delta } => args[index].push_str(&delta),
                        ToolEvent::CallEnd { call: Some(c), index, .. } => {
                            if c.arguments != json!({}) || !args[index].is_empty() {
                                assert_eq!(serde_json::from_str::<Value>(&args[index]).unwrap(), c.arguments, "{text}");
                            }
                            calls.push(c);
                        }
                        ToolEvent::CallEnd { call: None, raw, .. } => rest.push_str(&raw),
                    }
                }
                let rest = if calls.is_empty() { rest } else { rest.trim().to_string() };
                assert_eq!((rest, calls), whole, "{text} in pieces of {size}");
            }
        }
    }

    #[test]
    fn a_truncated_call_is_reported_incomplete() {
        let events = stream("<tool_call>{\"name\": \"a\", \"arguments\": {\"cut", 4, &["a"]);
        assert!(events.iter().any(|e| matches!(e, ToolEvent::CallStart { .. })));
        assert!(
            events.iter().any(|e| matches!(e, ToolEvent::CallEnd { call: None, .. })),
            "{events:?}"
        );
    }
}
