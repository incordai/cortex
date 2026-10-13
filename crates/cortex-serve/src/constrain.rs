//! Constrained decoding: restrict each step to the tokens that keep the
//! output valid.
//!
//! [`JsonConstraint`] accepts exactly the byte strings that are prefixes of a
//! JSON document matching a compiled [`Schema`] (any JSON, any object, or a
//! full JSON Schema; see `schema.rs` for the keywords enforced), and allows
//! the stop tokens only once the document is complete.
//!
//! The recognizer is a pushdown automaton over the schema graph. Its stack
//! is a persistent list (`Arc` links), so testing a token clones only the
//! top frame; unions (`anyOf`, type lists) run as parallel threads that die
//! as soon as a byte rules them out. Bounded numbers and lengths are checked
//! for viability on every byte, so the decoder never lets the model into a
//! state that cannot be completed.
//!
//! Callers: `engine.rs` (`Worker::sample_into` masks the logits with
//! [`Constraint::allowed`] and reports the choice with
//! [`Constraint::advance`]); the HTTP handlers in `api/` build constraints
//! for `response_format`, Ollama `format` and forced tool calls.

use std::sync::Arc;

use crate::model::Tokenizer;
use crate::schema::{
    ANY_ARRAY, ANY_NUMBER, ANY_OBJECT, ANY_STRING, ArrayNode, BOOLEAN, MAX_DEPTH, Node, NodeId, NULL, NumberNode, ObjectNode,
    Schema,
};

/// Decides which tokens may come next.
pub trait Constraint: Send {
    /// One flag per token id. `stop_tokens` should be allowed exactly when
    /// the output may end here.
    fn allowed(&self, stop_tokens: &[u32]) -> Vec<bool>;

    /// Records that `token` was produced.
    fn advance(&mut self, token: u32);
}

/// The bytes of every token of a vocabulary, built once per tokenizer.
#[derive(Debug)]
pub struct TokenTable {
    bytes: Vec<Option<Vec<u8>>>,
}

impl TokenTable {
    /// Reads every token's bytes from `tokenizer`.
    pub fn new(tokenizer: &dyn Tokenizer) -> Self {
        let bytes = (0..tokenizer.vocab_size() as u32).map(|t| tokenizer.token_bytes(t)).collect();
        Self { bytes }
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Whitespace run allowed between JSON tokens, so a model cannot stall the
/// output on endless blank space.
const MAX_WHITESPACE: u8 = 8;
/// Most alternatives tracked at once. Value starts usually decide the
/// alternative on their first byte; beyond this, the earliest alternatives
/// (in schema order) are kept.
const MAX_THREADS: usize = 32;
/// Longest number written, in characters.
const MAX_NUMBER: usize = 32;
/// Most fraction digits of a bounded number: within `f64` precision, so
/// every digit written can still change the value checked against the
/// bounds.
const MAX_FRACTION: usize = 15;
/// Longest object key, in bytes.
const MAX_KEY: u16 = 1024;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// Where a number's text stands in the JSON number grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumState {
    Empty,
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

/// A number being written: its grammar state and its text so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NumText {
    state: NumState,
    len: u8,
    text: [u8; MAX_NUMBER],
}

/// What a byte does to a number.
enum NumStep {
    /// It extends the number.
    Extended(NumText),
    /// It would extend the number but is not allowed (grammar, length, or
    /// no value within the bounds could follow).
    Refused,
    /// It is not part of the number; the number ends before it.
    Ends,
}

impl NumText {
    const EMPTY: Self = Self {
        state: NumState::Empty,
        len: 0,
        text: [0; MAX_NUMBER],
    };

    fn complete(&self) -> bool {
        matches!(self.state, NumState::Zero | NumState::Int | NumState::Frac | NumState::ExpDigits)
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.text[..self.len as usize]).expect("number text is ASCII")
    }

    fn step(&self, b: u8, n: &NumberNode) -> NumStep {
        use NumState::*;
        let fraction = !n.integer;
        let exponent = !n.integer && !n.bounded();
        let next = match (self.state, b) {
            (Empty, b'-') => Minus,
            (Empty | Minus, b'0') => Zero,
            (Empty | Minus | Int, b'1'..=b'9') | (Int, b'0') => Int,
            (Zero | Int, b'.') if fraction => Dot,
            (Dot | Frac, b'0'..=b'9') => Frac,
            (Zero | Int | Frac, b'e' | b'E') if exponent => Exp,
            (Exp, b'+' | b'-') => ExpSign,
            (Exp | ExpSign | ExpDigits, b'0'..=b'9') => ExpDigits,
            (_, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') => return NumStep::Refused,
            _ => return NumStep::Ends,
        };
        if self.len as usize == MAX_NUMBER {
            return NumStep::Refused;
        }
        if next == Frac && n.bounded() && self.fraction_digits() >= MAX_FRACTION {
            return NumStep::Refused;
        }
        let mut out = *self;
        out.state = next;
        out.text[out.len as usize] = b;
        out.len += 1;
        if n.bounded() && !viable(&out, n) {
            return NumStep::Refused;
        }
        NumStep::Extended(out)
    }

    fn fraction_digits(&self) -> usize {
        self.as_str().split_once('.').map_or(0, |(_, f)| f.len())
    }

    fn in_bounds(&self, n: &NumberNode) -> bool {
        !n.bounded() || self.as_str().parse::<f64>().is_ok_and(|v| point_in(v, n))
    }
}

/// Whether `v` meets `n`'s bounds.
fn point_in(v: f64, n: &NumberNode) -> bool {
    n.lower.is_none_or(|b| v > b.value || (v == b.value && !b.exclusive))
        && n.upper.is_none_or(|b| v < b.value || (v == b.value && !b.exclusive))
}

/// Whether some real number in `[lo, hi]` (each end closed or open) meets
/// `n`'s bounds.
fn overlaps(lo: f64, lo_closed: bool, hi: f64, hi_closed: bool, n: &NumberNode) -> bool {
    let (mut lo, mut lo_closed, mut hi, mut hi_closed) = (lo, lo_closed, hi, hi_closed);
    if let Some(b) = n.lower
        && (b.value > lo || (b.value == lo && b.exclusive))
    {
        lo = b.value;
        lo_closed = !b.exclusive;
    }
    if let Some(b) = n.upper
        && (b.value < hi || (b.value == hi && b.exclusive))
    {
        hi = b.value;
        hi_closed = !b.exclusive;
    }
    lo < hi || (lo == hi && lo_closed && hi_closed)
}

/// Whether some integer in `[lo, hi]` meets `n`'s bounds.
fn overlaps_integer(lo: f64, hi: f64, n: &NumberNode) -> bool {
    let lower = n
        .lower
        .map_or(f64::NEG_INFINITY, |b| if b.exclusive { b.value.floor() + 1.0 } else { b.value.ceil() });
    let upper = n
        .upper
        .map_or(f64::INFINITY, |b| if b.exclusive { b.value.ceil() - 1.0 } else { b.value.floor() });
    lo.max(lower) <= hi.min(upper)
}

/// Whether some completion of the (bounded, exponent-free) number prefix
/// `num` lies within `n`'s bounds.
fn viable(num: &NumText, n: &NumberNode) -> bool {
    let text = num.as_str();
    let negative = text.starts_with('-');
    let body = text.trim_start_matches('-');
    let (int_part, frac) = match body.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (body, None),
    };
    let room = (MAX_NUMBER - num.len as usize) as i32;
    // `[a, b)` of non-negative magnitudes, mirrored for negative numbers.
    let magnitudes = |a: f64, b: f64, b_closed: bool| {
        if negative {
            overlaps(-b, b_closed, -a, true, n)
        } else {
            overlaps(a, true, b, b_closed, n)
        }
    };
    if int_part.is_empty() {
        // Only "-": any non-positive number that still fits.
        let reach = 10f64.powi(room);
        return if n.integer {
            overlaps_integer(-reach, 0.0, n)
        } else {
            overlaps(-reach, false, 0.0, true, n)
        };
    }
    if let Some(frac) = frac {
        let p: f64 = format!("{int_part}.{frac}0").parse().unwrap_or(0.0);
        let signed = if negative { -p } else { p };
        // No digit may follow: the value is what it is.
        if frac.len() >= MAX_FRACTION || num.len as usize == MAX_NUMBER {
            return point_in(signed, n);
        }
        // Digits up to MAX_FRACTION can follow: from p up to just below the
        // next value of this precision.
        let top = p + 10f64.powi(-(frac.len() as i32)) - 10f64.powi(-(MAX_FRACTION as i32));
        return magnitudes(p, top, true);
    }
    if int_part == "0" {
        return if n.integer {
            overlaps_integer(0.0, 0.0, n)
        } else {
            magnitudes(0.0, 1.0, false)
        };
    }
    let v: f64 = int_part.parse().unwrap_or(0.0);
    for k in 0..=room.max(0) {
        let scale = 10f64.powi(k);
        let (a, b) = (v * scale, (v + 1.0) * scale);
        let hit = if n.integer {
            if negative {
                overlaps_integer(-(b - 1.0), -a, n)
            } else {
                overlaps_integer(a, b - 1.0, n)
            }
        } else {
            magnitudes(a, b, false)
        };
        if hit {
            return true;
        }
        // Longer digit strings only move further from zero.
        let past = if negative {
            n.lower.is_some_and(|l| -a < l.value)
        } else {
            n.upper.is_some_and(|u| a > u.value)
        };
        if past {
            return false;
        }
    }
    false
}

/// The escape-aware reader of a JSON string's content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Decoder {
    /// 0 plain, 1 after `\`, 2..=5 hex digits of `\u`, 6 a high surrogate
    /// wants `\`, 7 it wants `u`, 8..=11 hex digits of the low surrogate.
    state: u8,
    acc: u16,
    high: u16,
}

/// What one string byte means.
enum Decoded {
    /// Not valid here.
    Refused,
    /// Part of an escape still being read.
    Pending(Decoder),
    /// The closing quote.
    Close,
    /// A raw content byte (part of the UTF-8 text as written).
    Raw(Decoder),
    /// An escape finished, standing for this character.
    Char(Decoder, char),
}

impl Decoder {
    const NEW: Self = Self { state: 0, acc: 0, high: 0 };

    fn feed(self, b: u8) -> Decoded {
        let mut d = self;
        match self.state {
            0 => match b {
                b'"' => Decoded::Close,
                b'\\' => {
                    d.state = 1;
                    Decoded::Pending(d)
                }
                0x00..=0x1f => Decoded::Refused,
                _ => Decoded::Raw(d),
            },
            1 => {
                let c = match b {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        d.state = 2;
                        d.acc = 0;
                        return Decoded::Pending(d);
                    }
                    _ => return Decoded::Refused,
                };
                d.state = 0;
                Decoded::Char(d, c)
            }
            6 if b == b'\\' => {
                d.state = 7;
                Decoded::Pending(d)
            }
            7 if b == b'u' => {
                d.state = 8;
                d.acc = 0;
                Decoded::Pending(d)
            }
            6 | 7 => Decoded::Refused,
            s => {
                let Some(digit) = (b as char).to_digit(16) else {
                    return Decoded::Refused;
                };
                d.acc = (d.acc << 4) | digit as u16;
                if s != 5 && s != 11 {
                    d.state += 1;
                    return Decoded::Pending(d);
                }
                let unit = d.acc;
                if s == 5 {
                    match unit {
                        0xD800..=0xDBFF => {
                            d.state = 6;
                            d.high = unit;
                            Decoded::Pending(d)
                        }
                        0xDC00..=0xDFFF => Decoded::Refused,
                        _ => {
                            d.state = 0;
                            char::from_u32(u32::from(unit)).map_or(Decoded::Refused, |c| Decoded::Char(d, c))
                        }
                    }
                } else {
                    if !(0xDC00..=0xDFFF).contains(&unit) {
                        return Decoded::Refused;
                    }
                    let c = 0x10000 + ((u32::from(d.high) - 0xD800) << 10) + (u32::from(unit) - 0xDC00);
                    d.state = 0;
                    char::from_u32(c).map_or(Decoded::Refused, |c| Decoded::Char(d, c))
                }
            }
        }
    }
}

/// Hashes of the undeclared keys an object already has.
#[derive(Debug, PartialEq)]
struct Keys {
    hash: u64,
    next: Option<Arc<Keys>>,
}

fn has_key(keys: &Option<Arc<Keys>>, hash: u64) -> bool {
    let mut at = keys.as_deref();
    while let Some(k) = at {
        if k.hash == hash {
            return true;
        }
        at = k.next.as_deref();
    }
    false
}

/// Which member an object key named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Member {
    Declared(u16),
    Extra,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjPhase {
    /// After `{`: a key or `}`.
    First,
    /// After `,`: a key.
    Key,
    /// After a key: `:`.
    Colon(Member),
    /// After a member's value: `,` or `}`.
    After,
}

/// One level of the recognizer's stack.
#[derive(Clone, Debug, PartialEq)]
enum Frame {
    /// A value of `node` must start here.
    Value { node: NodeId },
    Object {
        node: NodeId,
        seen: u128,
        count: u32,
        phase: ObjPhase,
        extra: Option<Arc<Keys>>,
    },
    /// An object key being written; the object is the frame below.
    Key {
        lo: u16,
        hi: u16,
        pos: u16,
        hash: u64,
        dec: Decoder,
    },
    Array { node: NodeId, count: u32, first: bool },
    Str { node: NodeId, len: u32, dec: Decoder },
    Num { node: NodeId, num: NumText },
    /// One of `node`'s literal texts, `lo..hi` still matching `pos` bytes.
    Lit { node: NodeId, lo: u32, hi: u32, pos: u32 },
}

#[derive(Debug, PartialEq)]
struct Link {
    frame: Frame,
    next: Option<Arc<Link>>,
}

/// One way of reading the output so far.
#[derive(Clone, Debug, PartialEq)]
struct Thread {
    /// The innermost frame; `None` once the document is complete.
    top: Option<Frame>,
    below: Option<Arc<Link>>,
    depth: u16,
    space: u8,
}

impl Thread {
    fn root(node: NodeId) -> Self {
        Self {
            top: Some(Frame::Value { node }),
            below: None,
            depth: 1,
            space: 0,
        }
    }

    /// Pushes `frame` above the current top.
    fn push(&mut self, frame: Frame) -> bool {
        if usize::from(self.depth) >= MAX_DEPTH {
            return false;
        }
        if let Some(top) = self.top.take() {
            self.below = Some(Arc::new(Link {
                frame: top,
                next: self.below.take(),
            }));
        }
        self.top = Some(frame);
        self.depth += 1;
        true
    }

    /// With the top already taken: makes the frame below the top again.
    fn restore_parent(&mut self) {
        self.depth -= 1;
        match self.below.take() {
            Some(link) => {
                self.top = Some(link.frame.clone());
                self.below = link.next.clone();
            }
            None => self.top = None,
        }
    }

    /// Counts a whitespace byte; `false` past the allowed run.
    fn space(&mut self) -> bool {
        if self.space >= MAX_WHITESPACE {
            return false;
        }
        self.space += 1;
        true
    }

    fn complete(&self, schema: &Schema) -> bool {
        match &self.top {
            None => true,
            Some(Frame::Num { node, num }) if self.below.is_none() => {
                num.complete() && num.in_bounds(&number_node(schema, *node))
            }
            Some(Frame::Lit { node, lo, hi, pos }) if self.below.is_none() => {
                lo < hi && literals(schema, *node)[*lo as usize].len() == *pos as usize
            }
            _ => false,
        }
    }
}

fn object_node(schema: &Schema, id: NodeId) -> &ObjectNode {
    match schema.resolve(id) {
        Node::Object(o) => o,
        _ => unreachable!("object frames hold object nodes"),
    }
}

fn array_node(schema: &Schema, id: NodeId) -> &ArrayNode {
    match schema.resolve(id) {
        Node::Array(a) => a,
        _ => unreachable!("array frames hold array nodes"),
    }
}

fn number_node(schema: &Schema, id: NodeId) -> NumberNode {
    match schema.resolve(id) {
        Node::Number(n) => *n,
        _ => unreachable!("number frames hold number nodes"),
    }
}

fn literals(schema: &Schema, id: NodeId) -> &[Vec<u8>] {
    match schema.resolve(id) {
        Node::Literals(l) => l,
        _ => unreachable!("literal frames hold literal nodes"),
    }
}

fn string_bounds(schema: &Schema, id: NodeId) -> (u32, u32) {
    match schema.resolve(id) {
        Node::String { min, max } => (*min, *max),
        _ => unreachable!("string frames hold string nodes"),
    }
}

/// The items of `lits[lo..hi]` whose byte `pos` is `b` (they share their
/// first `pos` bytes and are sorted, so the result is a sub-range).
fn narrow(lits: &[Vec<u8>], lo: u32, hi: u32, pos: u32, b: u8) -> (u32, u32) {
    let mut start = hi;
    let mut end = hi;
    for i in lo..hi {
        let matches = lits[i as usize].get(pos as usize) == Some(&b);
        if matches && start == hi {
            start = i;
        }
        if !matches && start != hi {
            end = i;
            break;
        }
    }
    if start == hi { (hi, hi) } else { (start, end) }
}

/// Whether a value of `node` placed one level below a frame at `depth` can
/// be completed without exceeding [`MAX_DEPTH`].
fn fits(schema: &Schema, depth: u16, node: NodeId) -> bool {
    usize::from(depth) + usize::from(schema.min_depth[node as usize]) <= MAX_DEPTH
}

/// The declared members of `o` (an object frame at `depth`) whose values can
/// still be written there.
fn writable(schema: &Schema, o: &ObjectNode, depth: u16) -> u128 {
    (0..o.names.len())
        .filter(|&i| o.usable & (1 << i) != 0 && fits(schema, depth, o.values[i]))
        .fold(0, |mask, i| mask | (1 << i))
}

/// Whether `o` (at `depth`) may take an undeclared member here.
fn extra_fits(schema: &Schema, o: &ObjectNode, depth: u16) -> bool {
    o.additional.is_some_and(|a| fits(schema, depth, a))
}

/// Whether some declared member in `names[lo..hi]` is unused and writable.
fn available(lo: u16, hi: u16, seen: u128, writable: u128) -> bool {
    (lo..hi).any(|i| (writable & !seen) & (1 << i) != 0)
}

/// Whether another member may follow in `o` (at `depth`).
fn may_add(schema: &Schema, o: &ObjectNode, seen: u128, count: u32, depth: u16) -> bool {
    count < o.max
        && (extra_fits(schema, o, depth) || available(0, o.names.len() as u16, seen, writable(schema, o, depth)))
}

/// Whether the object may close now.
fn may_close(o: &ObjectNode, seen: u128, count: u32) -> bool {
    seen & o.required == o.required && count >= o.min
}

/// The schema of item `index` of an array.
fn item(a: &ArrayNode, index: u32) -> NodeId {
    a.prefix
        .get(index as usize)
        .copied()
        .or(a.items)
        .expect("max items never exceeds what the schema describes")
}

/// Starts a value of `node` with byte `b`; `t.top` is empty (the `Value`
/// frame was taken). Survivors go to `out`.
fn start(schema: &Schema, mut t: Thread, node: NodeId, b: u8, out: &mut Vec<Thread>) {
    let mut id = node;
    while let Node::Alias(next) = &schema.nodes[id as usize] {
        id = *next;
    }
    // The value replaces the `Value` frame at `t.depth`; it must be able to
    // finish within the depth left.
    if !fits(schema, t.depth - 1, id) {
        return;
    }
    let frame = match &schema.nodes[id as usize] {
        Node::Any => {
            let any = match b {
                b'{' => ANY_OBJECT,
                b'[' => ANY_ARRAY,
                b'"' => ANY_STRING,
                b'-' | b'0'..=b'9' => ANY_NUMBER,
                b't' | b'f' => BOOLEAN,
                b'n' => NULL,
                _ => return,
            };
            return start(schema, t, any, b, out);
        }
        Node::Union(alts) => {
            for &alt in alts {
                start(schema, t.clone(), alt, b, out);
            }
            return;
        }
        Node::Object(_) if b == b'{' => Frame::Object {
            node: id,
            seen: 0,
            count: 0,
            phase: ObjPhase::First,
            extra: None,
        },
        Node::Array(_) if b == b'[' => Frame::Array {
            node: id,
            count: 0,
            first: true,
        },
        Node::String { .. } if b == b'"' => Frame::Str {
            node: id,
            len: 0,
            dec: Decoder::NEW,
        },
        Node::Number(n) => match NumText::EMPTY.step(b, n) {
            NumStep::Extended(num) => Frame::Num { node: id, num },
            NumStep::Refused | NumStep::Ends => return,
        },
        Node::Literals(lits) => {
            let (lo, hi) = narrow(lits, 0, lits.len() as u32, 0, b);
            if lo == hi {
                return;
            }
            Frame::Lit { node: id, lo, hi, pos: 1 }
        }
        Node::Object(_) | Node::Array(_) | Node::String { .. } | Node::Alias(_) => return,
    };
    t.top = Some(frame);
    out.push(t);
}

/// Feeds byte `b` to thread `t`; every reading that survives goes to `out`.
fn step(schema: &Schema, mut t: Thread, b: u8, out: &mut Vec<Thread>) {
    loop {
        let Some(frame) = t.top.take() else {
            // The document is complete: nothing may follow.
            return;
        };
        match frame {
            Frame::Num { node, num } => match num.step(b, &number_node(schema, node)) {
                NumStep::Extended(num) => {
                    t.top = Some(Frame::Num { node, num });
                    out.push(t);
                    return;
                }
                NumStep::Refused => return,
                NumStep::Ends => {
                    if !(num.complete() && num.in_bounds(&number_node(schema, node))) {
                        return;
                    }
                    // The number ends here; `b` belongs to what follows it.
                    t.restore_parent();
                }
            },
            Frame::Lit { node, lo, hi, pos } => {
                let lits = literals(schema, node);
                let (nlo, nhi) = narrow(lits, lo, hi, pos, b);
                if nlo < nhi {
                    t.top = Some(Frame::Lit {
                        node,
                        lo: nlo,
                        hi: nhi,
                        pos: pos + 1,
                    });
                    out.push(t);
                    return;
                }
                if lits[lo as usize].len() != pos as usize {
                    return;
                }
                // A literal just completed (only numbers can end this way).
                t.restore_parent();
            }
            Frame::Str { node, len, dec } => {
                let (min, max) = string_bounds(schema, node);
                let (dec, len) = match dec.feed(b) {
                    Decoded::Refused => return,
                    Decoded::Close => {
                        if len < min {
                            return;
                        }
                        t.restore_parent();
                        out.push(t);
                        return;
                    }
                    Decoded::Pending(d) => (d, len),
                    // Count characters: every byte but UTF-8 continuations.
                    Decoded::Raw(d) => (d, if b & 0xC0 == 0x80 { len } else { len + 1 }),
                    Decoded::Char(d, _) => (d, len + 1),
                };
                if len > max {
                    return;
                }
                t.top = Some(Frame::Str { node, len, dec });
                out.push(t);
                return;
            }
            Frame::Key { lo, hi, pos, hash, dec } => {
                key_byte(schema, t, (lo, hi, pos, hash, dec), b, out);
                return;
            }
            Frame::Value { node } => {
                if is_space(b) {
                    t.top = Some(Frame::Value { node });
                    if t.space() {
                        out.push(t);
                    }
                    return;
                }
                t.space = 0;
                start(schema, t, node, b, out);
                return;
            }
            Frame::Object {
                node,
                seen,
                count,
                phase,
                extra,
            } => {
                let o = object_node(schema, node);
                if is_space(b) {
                    t.top = Some(Frame::Object {
                        node,
                        seen,
                        count,
                        phase,
                        extra,
                    });
                    if t.space() {
                        out.push(t);
                    }
                    return;
                }
                t.space = 0;
                let keep = |t: &mut Thread, phase, count| {
                    t.top = Some(Frame::Object {
                        node,
                        seen,
                        count,
                        phase,
                        extra: extra.clone(),
                    });
                };
                let depth = t.depth;
                match (phase, b) {
                    (ObjPhase::First | ObjPhase::Key, b'"') if may_add(schema, o, seen, count, depth) => {
                        keep(&mut t, phase, count);
                        let key = Frame::Key {
                            lo: 0,
                            hi: o.names.len() as u16,
                            pos: 0,
                            hash: FNV_OFFSET,
                            dec: Decoder::NEW,
                        };
                        if t.push(key) {
                            out.push(t);
                        }
                    }
                    (ObjPhase::First | ObjPhase::After, b'}') if may_close(o, seen, count) => {
                        t.restore_parent();
                        out.push(t);
                    }
                    (ObjPhase::After, b',') if may_add(schema, o, seen, count, depth) => {
                        keep(&mut t, ObjPhase::Key, count);
                        out.push(t);
                    }
                    (ObjPhase::Colon(member), b':') => {
                        let value = match member {
                            Member::Declared(i) => o.values[i as usize],
                            Member::Extra => o.additional.expect("extra keys need additionalProperties"),
                        };
                        keep(&mut t, ObjPhase::After, count + 1);
                        if t.push(Frame::Value { node: value }) {
                            out.push(t);
                        }
                    }
                    _ => {}
                }
                return;
            }
            Frame::Array { node, count, first } => {
                let a = array_node(schema, node);
                if is_space(b) {
                    t.top = Some(Frame::Array { node, count, first });
                    if t.space() {
                        out.push(t);
                    }
                    return;
                }
                t.space = 0;
                // Another item only if it can be finished below this frame.
                let more = count < a.max && fits(schema, t.depth, item(a, count));
                match b {
                    b']' if count >= a.min => {
                        t.restore_parent();
                        out.push(t);
                        return;
                    }
                    b',' if !first && more => {
                        t.top = Some(Frame::Array {
                            node,
                            count: count + 1,
                            first: false,
                        });
                        if t.push(Frame::Value { node: item(a, count) }) {
                            out.push(t);
                        }
                        return;
                    }
                    _ if first && more && b != b']' && b != b',' => {
                        // The first item starts with this very byte.
                        t.top = Some(Frame::Array {
                            node,
                            count: 1,
                            first: false,
                        });
                        if !t.push(Frame::Value { node: item(a, 0) }) {
                            return;
                        }
                    }
                    _ => return,
                }
            }
        }
    }
}

/// The first character of `bytes`, if they start with a whole one.
fn first_char(bytes: &[u8]) -> Option<char> {
    (1..=bytes.len().min(4)).find_map(|n| std::str::from_utf8(&bytes[..n]).ok().and_then(|s| s.chars().next()))
}

/// Whether the escape being read in a key (`d`, still pending) can decode
/// to the next character of some unused declared name in `names[lo..hi]`,
/// whose first `pos` bytes are already written.
fn escape_viable(d: Decoder, names: &[Vec<u8>], (lo, hi, pos): (u16, u16, u16), open: u128) -> bool {
    // `digits` hex digits of a UTF-16 unit written so far, equal to `acc`.
    let prefix = |unit: u16, digits: u8, acc: u16| digits == 0 || unit >> (16 - 4 * u16::from(digits)) == acc;
    (lo..hi).filter(|&i| open & (1 << i) != 0).any(|i| {
        let Some(c) = names[i as usize].get(pos as usize..).and_then(first_char) else {
            return false;
        };
        let mut buf = [0u16; 2];
        let units = c.encode_utf16(&mut buf);
        match d.state {
            1 => true,
            s @ 2..=5 => prefix(units[0], s - 2, d.acc),
            6 | 7 => units.len() == 2 && units[0] == d.high,
            s @ 8..=11 => units.len() == 2 && units[0] == d.high && prefix(units[1], s - 8, d.acc),
            _ => false,
        }
    })
}

/// Feeds byte `b` to an object key (`t.top` taken; the object is below).
fn key_byte(schema: &Schema, mut t: Thread, key: (u16, u16, u16, u64, Decoder), b: u8, out: &mut Vec<Thread>) {
    let (mut lo, mut hi, mut pos, mut hash, dec) = key;
    let Some(Frame::Object { node, seen, extra, .. }) = t.below.as_ref().map(|l| &l.frame) else {
        unreachable!("a key frame sits on its object")
    };
    let (node, seen, extra) = (*node, *seen, extra.clone());
    let o = object_node(schema, node);
    // The members whose values can still be written below the object (one
    // frame under this key).
    let object_depth = t.depth - 1;
    let open = writable(schema, o, object_depth) & !seen;
    let extra_ok = extra_fits(schema, o, object_depth);
    let feed = |c: u8, lo: &mut u16, hi: &mut u16, pos: &mut u16, hash: &mut u64| -> bool {
        if *pos >= MAX_KEY {
            return false;
        }
        let (a, z) = narrow(&o.names, u32::from(*lo), u32::from(*hi), u32::from(*pos), c);
        (*lo, *hi) = (a as u16, z as u16);
        *pos += 1;
        *hash = (*hash ^ u64::from(c)).wrapping_mul(FNV_PRIME);
        extra_ok || available(*lo, *hi, 0, open)
    };
    let dec = match dec.feed(b) {
        Decoded::Refused => return,
        // An escape in progress must still be able to spell a declared name
        // (any key is fine when undeclared members are allowed).
        Decoded::Pending(d) => {
            if !extra_ok && !escape_viable(d, &o.names, (lo, hi, pos), open) {
                return;
            }
            d
        }
        Decoded::Raw(d) => {
            if !feed(b, &mut lo, &mut hi, &mut pos, &mut hash) {
                return;
            }
            d
        }
        Decoded::Char(d, c) => {
            let mut buf = [0u8; 4];
            for &byte in c.encode_utf8(&mut buf).as_bytes() {
                if !feed(byte, &mut lo, &mut hi, &mut pos, &mut hash) {
                    return;
                }
            }
            d
        }
        Decoded::Close => {
            let exact = lo < hi && o.names[lo as usize].len() == pos as usize;
            let mut extra = extra.clone();
            let (member, seen) = if exact {
                if open & (1 << lo) == 0 {
                    return; // already written, or its value cannot be written here
                }
                (Member::Declared(lo), seen | (1 << lo))
            } else {
                if !extra_ok || has_key(&extra, hash) {
                    return;
                }
                extra = Some(Arc::new(Keys { hash, next: extra }));
                (Member::Extra, seen)
            };
            t.restore_parent();
            let Some(Frame::Object { count, .. }) = t.top else {
                unreachable!("restored the object")
            };
            t.top = Some(Frame::Object {
                node,
                seen,
                count,
                phase: ObjPhase::Colon(member),
                extra,
            });
            out.push(t);
            return;
        }
    };
    t.top = Some(Frame::Key { lo, hi, pos, hash, dec });
    out.push(t);
}

/// Feeds `b` to every thread of `from` into `to` (cleared first), dropping
/// duplicates.
fn advance_all(schema: &Schema, from: &mut Vec<Thread>, to: &mut Vec<Thread>, b: u8) {
    to.clear();
    for t in from.drain(..) {
        step(schema, t, b, to);
    }
    if to.len() > 1 {
        let mut unique: Vec<Thread> = Vec::with_capacity(to.len());
        for t in to.drain(..) {
            if !unique.contains(&t) {
                unique.push(t);
            }
        }
        unique.truncate(MAX_THREADS);
        *to = unique;
    }
}

/// Output restricted to JSON matching a [`Schema`].
#[derive(Clone, Debug)]
pub struct JsonConstraint {
    table: Arc<TokenTable>,
    schema: Arc<Schema>,
    threads: Vec<Thread>,
}

impl JsonConstraint {
    /// Any JSON value, or with `root_object` any JSON object
    /// (`response_format: json_object`).
    pub fn new(table: Arc<TokenTable>, root_object: bool) -> Self {
        let schema = if root_object { Schema::any_object() } else { Schema::any_value() };
        Self::with_schema(table, Arc::new(schema))
    }

    /// JSON matching `schema`.
    pub fn with_schema(table: Arc<TokenTable>, schema: Arc<Schema>) -> Self {
        let threads = vec![Thread::root(schema.root)];
        Self { table, schema, threads }
    }

    /// Feeds raw bytes, as if a token with these bytes had been produced.
    /// Returns `false` (leaving the state unchanged) if they cannot continue
    /// a matching document.
    pub fn feed(&mut self, bytes: &[u8]) -> bool {
        let mut a = self.threads.clone();
        let mut b = Vec::new();
        for &byte in bytes {
            advance_all(&self.schema, &mut a, &mut b, byte);
            if b.is_empty() {
                return false;
            }
            std::mem::swap(&mut a, &mut b);
        }
        self.threads = a;
        true
    }

    /// Whether a complete document has been produced.
    pub fn is_complete(&self) -> bool {
        self.threads.iter().any(|t| t.complete(&self.schema))
    }

    fn accepts(&self, bytes: &[u8], a: &mut Vec<Thread>, b: &mut Vec<Thread>) -> bool {
        if bytes.is_empty() {
            return false;
        }
        a.clear();
        a.extend(self.threads.iter().cloned());
        for &byte in bytes {
            advance_all(&self.schema, a, b, byte);
            if b.is_empty() {
                return false;
            }
            std::mem::swap(a, b);
        }
        true
    }
}

impl Constraint for JsonConstraint {
    fn allowed(&self, stop_tokens: &[u32]) -> Vec<bool> {
        let complete = self.is_complete();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut out: Vec<bool> = self
            .table
            .bytes
            .iter()
            .map(|bytes| bytes.as_deref().is_some_and(|bytes| self.accepts(bytes, &mut a, &mut b)))
            .collect();
        for &t in stop_tokens {
            if let Some(slot) = out.get_mut(t as usize) {
                *slot = complete;
            }
        }
        out
    }

    fn advance(&mut self, token: u32) {
        let bytes = self
            .table
            .bytes
            .get(token as usize)
            .cloned()
            .flatten()
            .unwrap_or_default();
        let accepted = self.feed(&bytes);
        debug_assert!(accepted, "a masked-out token was produced");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn empty_table() -> Arc<TokenTable> {
        Arc::new(TokenTable { bytes: Vec::new() })
    }

    fn constraint(root_object: bool) -> JsonConstraint {
        JsonConstraint::new(empty_table(), root_object)
    }

    fn accepts_whole(text: &str, root_object: bool) -> (bool, bool) {
        let mut c = constraint(root_object);
        let ok = c.feed(text.as_bytes());
        (ok, c.is_complete())
    }

    fn schema(s: Value) -> JsonConstraint {
        JsonConstraint::with_schema(empty_table(), Arc::new(Schema::compile(&s).unwrap()))
    }

    /// `(prefix accepted, complete)` for `text` under schema `s`.
    fn check(s: &Value, text: &str) -> (bool, bool) {
        let mut c = schema(s.clone());
        let ok = c.feed(text.as_bytes());
        (ok, ok && c.is_complete())
    }

    #[test]
    fn complete_documents_are_accepted_and_complete() {
        for doc in [
            r#"{"a": [1, -2.5e+3, true, null, "x\"yé"], "b": {}}"#,
            "[]",
            "0",
            "\"s\"",
            "-0.5",
            "{\"k\":[[],{}]}",
        ] {
            assert_eq!(accepts_whole(doc, false), (true, true), "{doc}");
        }
    }

    #[test]
    fn prefixes_are_accepted_but_not_complete() {
        for doc in [r#"{"a": [1, "#, "[tr", r#"{"k""#, "-", "1.", "1e", r#""ab\u00"#] {
            assert_eq!(accepts_whole(doc, false), (true, false), "{doc}");
        }
    }

    #[test]
    fn invalid_bytes_are_refused_without_changing_state() {
        for doc in ["{a", "[1,]", "01", "tru e", "{\"a\" 1}", "1.e5", "[1}", "\"\u{1}\"", "{} {}", "[,1]"] {
            assert!(!accepts_whole(doc, false).0, "{doc}");
        }
        let mut c = constraint(false);
        assert!(c.feed(b"[1"));
        assert!(!c.feed(b"}"));
        assert!(c.feed(b"]"));
        assert!(c.is_complete());
    }

    #[test]
    fn root_object_mode_refuses_other_roots() {
        assert!(!accepts_whole("[1]", true).0);
        assert!(!accepts_whole("1", true).0);
        assert_eq!(accepts_whole("{\"a\":1}", true), (true, true));
    }

    #[test]
    fn a_number_ends_at_a_closer_inside_containers() {
        assert_eq!(accepts_whole("[10]", false), (true, true));
        assert_eq!(accepts_whole("{\"a\":1.5}", false), (true, true));
        assert_eq!(accepts_whole("[1 ,2]", false), (true, true));
    }

    #[test]
    fn whitespace_runs_are_bounded() {
        let long = format!("[{}1]", " ".repeat(MAX_WHITESPACE as usize + 1));
        assert!(!accepts_whole(&long, false).0);
    }

    #[test]
    fn surrogate_escapes_must_pair() {
        assert_eq!(accepts_whole(r#""😀""#, false), (true, true));
        assert!(!accepts_whole(r#""\ud83d x""#, false).0);
        assert!(!accepts_whole(r#""\ude00""#, false).0);
    }

    #[test]
    fn object_members_follow_the_schema() {
        let s = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 4},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
            },
            "required": ["name", "age"],
            "additionalProperties": false,
        });
        assert_eq!(check(&s, r#"{"name": "Ann", "age": 30, "tags": ["a"]}"#), (true, true));
        assert_eq!(check(&s, r#"{"age":-1,"name":"Bo"}"#), (true, true), "any member order");
        assert!(!check(&s, r#"{"name": "Ann"}"#).1, "age is required");
        assert!(!check(&s, r#"{"name": "Ann", "x""#).0, "undeclared members are refused at the key");
        assert!(!check(&s, r#"{"nom""#).0);
        assert!(!check(&s, r#"{"name": "A""#).0, "too short");
        assert!(!check(&s, r#"{"name": "Annie"#).0, "too long");
        assert!(!check(&s, r#"{"age": 1.5"#).0, "integers have no fraction");
        assert!(!check(&s, r#"{"age": 1, "age""#).0, "no member twice");
        assert!(!check(&s, r#"{"tags": ["c""#).0);
        assert!(!check(&s, r#"{"tags": ["a", "b", "#).0, "maxItems");
        assert!(!check(&s, r#"{"name": "Ann", "age": 1}}"#).0);
        // `}` is refused while a required member is missing.
        assert!(!check(&s, r#"{"name": "Ann"}"#).0);
    }

    #[test]
    fn escaped_keys_match_their_declared_names() {
        let s = json!({"type": "object", "properties": {"é": {"type": "null"}}, "required": ["é"], "additionalProperties": false});
        assert_eq!(check(&s, r#"{"é": null}"#), (true, true));
        assert_eq!(check(&s, r#"{"é": null}"#), (true, true));
    }

    #[test]
    fn additional_members_take_their_schema_and_cannot_repeat() {
        let s = json!({"type": "object", "properties": {"a": {"type": "boolean"}}, "additionalProperties": {"type": "integer"}});
        assert_eq!(check(&s, r#"{"a": true, "b": 2, "c": 3}"#), (true, true));
        assert!(!check(&s, r#"{"b": "x""#).0);
        assert!(!check(&s, r#"{"b": 1, "b""#).0);
        assert!(!check(&s, r#"{"a": 1"#).0, "a declared key uses its own schema");
    }

    #[test]
    fn enums_consts_and_unions() {
        let colour = json!({"enum": ["red", "green", 7, null]});
        for ok in [r#""red""#, r#""green""#, "7", "null"] {
            assert_eq!(check(&colour, ok), (true, true), "{ok}");
        }
        for bad in [r#""blue""#, "8", r#""re"#] {
            assert!(!check(&colour, bad).1, "{bad}");
        }
        assert!(!check(&colour, r#""b"#).0);
        let numbers = json!({"enum": [1, 12]});
        assert_eq!(check(&numbers, "1"), (true, true));
        assert_eq!(check(&numbers, "12"), (true, true));
        let in_array = json!({"type": "array", "items": {"enum": [1, 12]}});
        assert_eq!(check(&in_array, "[1,12]"), (true, true));
        assert!(!check(&in_array, "[13").0);

        let union = json!({"anyOf": [
            {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}, "required": ["kind", "x"], "additionalProperties": false},
            {"type": "object", "properties": {"kind": {"const": "b"}, "y": {"type": "string"}}, "required": ["kind", "y"], "additionalProperties": false},
        ]});
        assert_eq!(check(&union, r#"{"kind": "a", "x": 1}"#), (true, true));
        assert_eq!(check(&union, r#"{"kind": "b", "y": "s"}"#), (true, true));
        assert!(!check(&union, r#"{"kind": "a", "y""#).0, "the alternative was decided by kind");
        assert_eq!(check(&json!({"type": ["string", "null"]}), "null"), (true, true));
    }

    #[test]
    fn numeric_bounds_are_enforced_on_every_prefix() {
        let s = json!({"type": "integer", "minimum": 10, "maximum": 250});
        for ok in ["10", "26", "99", "250", "123"] {
            assert_eq!(check(&s, ok), (true, true), "{ok}");
        }
        // Prefixes no completion of which is in range are refused outright.
        for bad in ["-", "0", "251", "260", "2501"] {
            assert!(!check(&s, bad).0, "{bad}");
        }
        // "3" is below 10 but 30..=39 are reachable: allowed, not complete.
        assert_eq!(check(&s, "3"), (true, false));
    }

    #[test]
    fn bounded_numbers_end_only_in_range() {
        let s = json!({"type": "number", "exclusiveMinimum": 0, "maximum": 1});
        assert_eq!(check(&s, "0.5"), (true, true));
        assert_eq!(check(&s, "1"), (true, true));
        assert!(!check(&s, "0").1, "0 is excluded");
        assert!(check(&s, "0.").0, "0.x can still be in range");
        assert!(!check(&s, "2").0, "every completion of 2 exceeds 1");
        assert!(!check(&s, "1.5").0);
        assert!(!check(&s, "1e").0, "bounded numbers are written without exponents");
        let in_array = json!({"type": "array", "items": s});
        assert_eq!(check(&in_array, "[0.25, 1]"), (true, true));
        assert!(!check(&in_array, "[0]").0, "the closer is refused after an out-of-range number");
    }

    #[test]
    fn recursive_schemas_nest() {
        let s = json!({
            "$defs": {"tree": {"type": "object", "properties": {"v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/tree"}}}, "required": ["v"], "additionalProperties": false}},
            "$ref": "#/$defs/tree",
        });
        assert_eq!(check(&s, r#"{"v": 1, "kids": [{"v": 2, "kids": [{"v": 3}]}]}"#), (true, true));
        assert!(!check(&s, r#"{"v": 1, "kids": [{"w""#).0);
    }

    #[test]
    fn tuples_and_item_counts() {
        let s = json!({"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": false, "minItems": 2});
        assert_eq!(check(&s, r#"["a", 1]"#), (true, true));
        assert!(!check(&s, r#"["a"]"#).0, "minItems");
        assert!(!check(&s, r#"["a", 1, "#).0, "no items past the prefix");
        assert!(!check(&s, "[1").0, "the first item is a string");
    }

    #[test]
    fn the_token_mask_follows_the_recognizer() {
        let table = Arc::new(TokenTable {
            bytes: vec![
                Some(b"{".to_vec()),
                Some(b"\"a\"".to_vec()),
                Some(b":1}".to_vec()),
                Some(b"x".to_vec()),
                None,
            ],
        });
        let mut c = JsonConstraint::new(table, true);
        assert_eq!(c.allowed(&[4]), vec![true, false, false, false, false]);
        c.advance(0);
        assert_eq!(c.allowed(&[4]), vec![false, true, false, false, false]);
        c.advance(1);
        c.advance(2);
        assert!(c.is_complete());
        assert_eq!(c.allowed(&[4]), vec![false, false, false, false, true], "only the stop token");
    }

    /// Every byte is a token; 256 stops.
    fn byte_table() -> Arc<TokenTable> {
        Arc::new(TokenTable {
            bytes: (0..=255u8).map(|b| Some(vec![b])).chain([None]).collect(),
        })
    }

    fn walk_schemas() -> Vec<Value> {
        vec![
            json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": false}),
            json!({"type": "object", "properties": {
                "colour": {"enum": ["red", "green"]}, "n": {"type": "integer", "minimum": -3, "maximum": 12},
                "x": {"type": "number", "exclusiveMinimum": 0.5, "maximum": 2}, "s": {"type": "string", "minLength": 1, "maxLength": 3},
                "é": {"type": "null"},
            }, "required": ["colour", "n", "x", "s", "é"], "additionalProperties": false}),
            json!({"anyOf": [
                {"type": "object", "properties": {"k": {"const": "a"}, "v": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}}, "required": ["k", "v"], "additionalProperties": false},
                {"type": "object", "properties": {"k": {"const": "b"}}, "required": ["k"], "additionalProperties": false},
            ]}),
            json!({"$defs": {"t": {"type": "object", "properties": {"kid": {"anyOf": [{"$ref": "#/$defs/t"}, {"type": "null"}]}}, "required": ["kid"], "additionalProperties": false}}, "$ref": "#/$defs/t"}),
        ]
    }

    #[test]
    fn byte_level_walks_never_reach_a_dead_end() {
        let table = byte_table();
        let stop = 256u32;
        for s in walk_schemas() {
            let compiled = Arc::new(Schema::compile(&s).unwrap());
            // Lowest byte first steers into escapes and whitespace; it need
            // not finish, but every step must leave some token allowed.
            let mut c = JsonConstraint::with_schema(table.clone(), compiled.clone());
            let mut text = Vec::new();
            for _ in 0..300 {
                let mask = c.allowed(&[stop]);
                let pick = mask
                    .iter()
                    .position(|&a| a)
                    .unwrap_or_else(|| panic!("dead end after {:?} under {s}", String::from_utf8_lossy(&text)));
                if pick as u32 == stop {
                    break;
                }
                text.push(pick as u8);
                c.advance(pick as u32);
            }
            // Structure first: must finish with a document the schema accepts
            // (recursion included: past MAX_DEPTH nesting is refused, so the
            // walk is forced to close).
            let mut c = JsonConstraint::with_schema(table.clone(), compiled);
            let mut text = Vec::new();
            let preference = b"\"}]:,{[0123456789.tfnrudgeabkvlsx-";
            for _ in 0..4000 {
                let mask = c.allowed(&[stop]);
                if mask[stop as usize] {
                    break;
                }
                let pick = preference
                    .iter()
                    .map(|&b| b as usize)
                    .find(|&b| mask[b])
                    .or_else(|| mask.iter().position(|&a| a))
                    .unwrap_or_else(|| panic!("dead end after {:?} under {s}", String::from_utf8_lossy(&text)));
                text.push(pick as u8);
                c.advance(pick as u32);
            }
            assert!(c.is_complete(), "unfinished {:?} under {s}", String::from_utf8_lossy(&text));
            serde_json::from_slice::<Value>(&text).unwrap_or_else(|e| panic!("{e}: {:?}", String::from_utf8_lossy(&text)));
        }
    }

    #[test]
    fn every_masked_path_parses_and_validates() {
        // A walk that always takes the first allowed token must end in a
        // document serde accepts, never in a dead end.
        let s = json!({
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 5, "maximum": 7}, "s": {"type": "string", "maxLength": 3}},
            "required": ["n", "s"],
            "additionalProperties": false,
        });
        let pieces: Vec<&[u8]> = vec![b"{", b"\"", b"n", b"s", b"\":", b"1", b"6", b"9", b",", b"x", b"}", b" ", b"\"s\":", b":"];
        let table = Arc::new(TokenTable {
            bytes: pieces.iter().map(|p| Some(p.to_vec())).chain([None]).collect(),
        });
        let stop = (pieces.len()) as u32;
        let mut c = JsonConstraint::with_schema(table.clone(), Arc::new(Schema::compile(&s).unwrap()));
        let mut text = Vec::new();
        for _ in 0..64 {
            let mask = c.allowed(&[stop]);
            let pick = mask
                .iter()
                .position(|&a| a)
                .unwrap_or_else(|| panic!("dead end after {:?}", String::from_utf8_lossy(&text)));
            if pick as u32 == stop {
                break;
            }
            text.extend_from_slice(pieces[pick]);
            c.advance(pick as u32);
        }
        let v: Value = serde_json::from_slice(&text).expect("valid JSON");
        let n = v["n"].as_i64().expect("n is an integer");
        assert!((5..=7).contains(&n), "{v}");
        assert!(v["s"].as_str().is_some_and(|s| s.chars().count() <= 3), "{v}");
    }
}
