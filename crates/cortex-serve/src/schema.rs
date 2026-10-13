//! JSON Schema compiled into the grammar the constrained decoder enforces.
//!
//! [`Schema::compile`] turns a JSON Schema document into a graph of
//! [`Node`]s; `constrain.rs` walks that graph byte by byte while the model
//! writes. Every keyword that changes which documents are valid is either
//! enforced exactly or refused with a [`SchemaError`] naming it, so a
//! request never gets output that silently ignores part of its schema.
//!
//! Enforced: `type` (one or a list), `enum`, `const`, `properties`,
//! `required`, `additionalProperties`, `minProperties`, `maxProperties`,
//! `items`, `prefixItems`, `minItems`, `maxItems`, `minLength`,
//! `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`,
//! `exclusiveMaximum` (number and draft-4 boolean forms), `anyOf`, `oneOf`
//! (as `anyOf`: the output matches at least one alternative), `allOf` with a
//! single member, `$ref` to `#`, `#/$defs/...` or `#/definitions/...`
//! (recursion allowed), `nullable` (OpenAPI). Annotations (`title`,
//! `description`, `default`, `examples`, `format`, ...) are ignored, as the
//! JSON Schema specification allows for `format`.
//!
//! Callers: `api/mod.rs` (`response_format`, Ollama `format`, forced tool
//! calls) and `constrain.rs`, which reads the nodes.

use std::collections::HashMap;
use std::fmt;

use serde_json::{Map, Value};

/// Index of a node in [`Schema::nodes`].
pub(crate) type NodeId = u32;

/// Any JSON value.
pub(crate) const ANY: NodeId = 0;
/// `true` or `false`.
pub(crate) const BOOLEAN: NodeId = 1;
/// `null`.
pub(crate) const NULL: NodeId = 2;
/// Any string.
pub(crate) const ANY_STRING: NodeId = 3;
/// Any number.
pub(crate) const ANY_NUMBER: NodeId = 4;
/// An object with any members.
pub(crate) const ANY_OBJECT: NodeId = 5;
/// An array of any values.
pub(crate) const ANY_ARRAY: NodeId = 6;

/// Most declared properties one object schema may have (the decoder tracks
/// them in a 128-bit set).
pub const MAX_PROPERTIES: usize = 128;
/// Most nodes one compiled schema may have.
const MAX_NODES: usize = 20_000;
/// Deepest schema nesting compiled (bounds the compiler's recursion).
const MAX_SCHEMA_DEPTH: usize = 128;

/// One numeric bound: the value and whether it is exclusive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bound {
    pub value: f64,
    pub exclusive: bool,
}

/// What a node accepts.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Node {
    /// Any JSON value.
    Any,
    /// An object.
    Object(ObjectNode),
    /// An array.
    Array(ArrayNode),
    /// A string of `min..=max` characters.
    String { min: u32, max: u32 },
    /// A number.
    Number(NumberNode),
    /// Exactly one of these JSON texts (compact serializations, sorted).
    Literals(Vec<Vec<u8>>),
    /// Any of these alternatives.
    Union(Vec<NodeId>),
    /// The same as another node (a resolved `$ref`).
    Alias(NodeId),
}

/// An object schema.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ObjectNode {
    /// Declared member names (unescaped UTF-8), sorted bytewise.
    pub names: Vec<Vec<u8>>,
    /// Schema of each declared member, aligned with `names`.
    pub values: Vec<NodeId>,
    /// Bit `i` set: `names[i]` is required.
    pub required: u128,
    /// Bit `i` set: `names[i]`'s schema admits a finite document, so the
    /// member can be written (see [`prune`]).
    pub usable: u128,
    /// Schema of undeclared members, `None` if they are not allowed.
    pub additional: Option<NodeId>,
    /// Fewest members.
    pub min: u32,
    /// Most members.
    pub max: u32,
}

/// An array schema.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ArrayNode {
    /// Schemas of the leading items, by position.
    pub prefix: Vec<NodeId>,
    /// Schema of the items after `prefix`, `None` if there may be none.
    pub items: Option<NodeId>,
    /// Fewest items.
    pub min: u32,
    /// Most items.
    pub max: u32,
}

/// A number schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NumberNode {
    /// Only integers (written without a fraction or exponent).
    pub integer: bool,
    /// Lower bound.
    pub lower: Option<Bound>,
    /// Upper bound.
    pub upper: Option<Bound>,
}

impl NumberNode {
    /// Whether the number is bounded (and so written without an exponent,
    /// which keeps every prefix checkable).
    pub fn bounded(&self) -> bool {
        self.lower.is_some() || self.upper.is_some()
    }
}

/// A compiled schema.
#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
    pub(crate) nodes: Vec<Node>,
    pub(crate) root: NodeId,
    /// Fewest stack frames a document matching each node needs (see
    /// [`prune`]).
    pub(crate) min_depth: Vec<u16>,
}

/// Why a schema cannot be enforced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError(String);

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SchemaError {}

fn err<T>(message: impl Into<String>) -> Result<T, SchemaError> {
    Err(SchemaError(message.into()))
}

/// Keywords that would change which documents are valid but are not
/// enforced; a schema using one is refused.
const UNSUPPORTED: &[&str] = &[
    "pattern",
    "patternProperties",
    "propertyNames",
    "multipleOf",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "unevaluatedItems",
    "additionalItems",
    "contentSchema",
    "$dynamicRef",
    "$recursiveRef",
];

/// Keywords that only describe or organize; they never change validity.
const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$anchor",
    "$comment",
    "$defs",
    "definitions",
    "$vocabulary",
    "title",
    "description",
    "default",
    "examples",
    "example",
    "format",
    "deprecated",
    "readOnly",
    "writeOnly",
    "contentEncoding",
    "contentMediaType",
    "discriminator",
    "strict",
];

/// The fixed nodes every schema starts with (ids [`ANY`]..=[`ANY_ARRAY`]).
fn builtins() -> Vec<Node> {
    vec![
        Node::Any,
        Node::Literals(vec![b"false".to_vec(), b"true".to_vec()]),
        Node::Literals(vec![b"null".to_vec()]),
        Node::String { min: 0, max: u32::MAX },
        Node::Number(NumberNode {
            integer: false,
            lower: None,
            upper: None,
        }),
        Node::Object(ObjectNode {
            names: Vec::new(),
            values: Vec::new(),
            required: 0,
            usable: 0,
            additional: Some(ANY),
            min: 0,
            max: u32::MAX,
        }),
        Node::Array(ArrayNode {
            prefix: Vec::new(),
            items: Some(ANY),
            min: 0,
            max: u32::MAX,
        }),
    ]
}

impl Schema {
    /// Any JSON document.
    pub fn any_value() -> Self {
        Self::builtin(ANY)
    }

    /// Any JSON object (`response_format: json_object`).
    pub fn any_object() -> Self {
        Self::builtin(ANY_OBJECT)
    }

    fn builtin(root: NodeId) -> Self {
        let mut nodes = builtins();
        let min_depth = prune(&mut nodes, root).expect("the built-in schemas are finite");
        Self { nodes, root, min_depth }
    }

    /// Compiles a JSON Schema document.
    ///
    /// # Errors
    /// If the schema uses a keyword that is not enforced, is malformed, or
    /// accepts no document at all.
    pub fn compile(schema: &Value) -> Result<Self, SchemaError> {
        let mut compiler = Compiler {
            root: schema,
            nodes: builtins(),
            refs: HashMap::new(),
            closed: false,
        };
        let root = compiler.node(schema, 0)?;
        let mut nodes = compiler.nodes;
        let min_depth = prune(&mut nodes, root)?;
        Ok(Self { nodes, root, min_depth })
    }

    /// A call to one of `tools` (name, parameters schema), written as the
    /// bare object `{"name": ..., "arguments": {...}}` that
    /// `tools::ToolParser` reads. Each parameters schema compiles against
    /// its own root, so its `$ref`s resolve as written; a `null` schema
    /// means any object. Objects in the parameters take only their declared
    /// members unless `additionalProperties` says otherwise (as in strict
    /// function calling: a function accepts the parameters it declares).
    ///
    /// # Errors
    /// If `tools` is empty or a parameters schema cannot be enforced.
    pub fn tool_calls(tools: &[(&str, &Value)]) -> Result<Self, SchemaError> {
        if tools.is_empty() {
            return err("no tool to call");
        }
        let any_object = Value::Object(Map::from_iter([("type".to_string(), Value::from("object"))]));
        let mut nodes = builtins();
        let mut alternatives = Vec::with_capacity(tools.len());
        for (name, params) in tools {
            let params = if params.is_null() { &any_object } else { *params };
            let mut compiler = Compiler {
                root: params,
                nodes: std::mem::take(&mut nodes),
                refs: HashMap::new(),
                closed: true,
            };
            let arguments = compiler
                .node(params, 0)
                .map_err(|e| SchemaError(format!("tool '{name}': {e}")))?;
            let name = compiler.push(Node::Literals(vec![
                serde_json::to_vec(&Value::from(*name)).expect("a string serializes"),
            ]))?;
            alternatives.push(compiler.push(Node::Object(ObjectNode {
                names: vec![b"arguments".to_vec(), b"name".to_vec()],
                values: vec![arguments, name],
                required: 0b11,
                usable: 0b11,
                additional: None,
                min: 0,
                max: u32::MAX,
            }))?);
            nodes = compiler.nodes;
        }
        let root = if alternatives.len() == 1 {
            alternatives[0]
        } else {
            nodes.push(Node::Union(alternatives));
            (nodes.len() - 1) as NodeId
        };
        let min_depth = prune(&mut nodes, root)?;
        Ok(Self { nodes, root, min_depth })
    }

    /// The node `id` stands for, following aliases.
    pub(crate) fn resolve(&self, mut id: NodeId) -> &Node {
        loop {
            match &self.nodes[id as usize] {
                Node::Alias(next) => id = *next,
                node => return node,
            }
        }
    }
}

/// Deepest nesting the decoder tracks (stack frames: one per open value or
/// key); deeper documents are refused.
pub(crate) const MAX_DEPTH: usize = 128;

/// Computes each node's minimum nesting depth and removes every way into a
/// node that cannot be completed within [`MAX_DEPTH`], so the decoder can
/// always finish what it starts. Returns the depths.
///
/// The depth of a node is the fewest stack frames a document matching it
/// needs: a scalar 1; an object 1, plus its deepest required member when it
/// has one (its key and value sit one level down); an array 1 plus its
/// deepest of the first `min` items; a union its shallowest alternative.
/// Computed by relaxation from "unreachable", so a required member that
/// recurses forever stays unreachable. Optional members, additional
/// members, union alternatives and items that cannot fit are dropped; the
/// decoder checks the remaining depth at run time for the rest. Refused if
/// the root itself cannot fit.
fn prune(nodes: &mut [Node], root: NodeId) -> Result<Vec<u16>, SchemaError> {
    const UNREACHABLE: u32 = u32::MAX;
    let mut depth = vec![UNREACHABLE; nodes.len()];
    let item = |a: &ArrayNode, i: u32| a.prefix.get(i as usize).copied().or(a.items);
    loop {
        let mut changed = false;
        for (i, node) in nodes.iter().enumerate() {
            let below = |deepest: u32| if deepest == UNREACHABLE { UNREACHABLE } else { deepest + 1 };
            let now = match node {
                Node::Any | Node::String { .. } | Node::Number(_) | Node::Literals(_) => 1,
                Node::Alias(t) => depth[*t as usize],
                Node::Union(alts) => alts.iter().map(|&a| depth[a as usize]).min().unwrap_or(UNREACHABLE),
                Node::Object(o) if o.required == 0 => 1,
                Node::Object(o) => below(
                    (0..o.names.len())
                        .filter(|&m| o.required & (1 << m) != 0)
                        .map(|m| depth[o.values[m] as usize])
                        .max()
                        .unwrap_or(1),
                ),
                Node::Array(a) if a.min == 0 => 1,
                Node::Array(a) => below((0..a.min).map(|k| item(a, k).map_or(UNREACHABLE, |n| depth[n as usize])).max().unwrap_or(1)),
            };
            if now < depth[i] {
                depth[i] = now;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let fits = |id: NodeId| depth[id as usize] as usize <= MAX_DEPTH;
    if !fits(root) {
        return err(format!(
            "the schema admits no document within {MAX_DEPTH} levels of nesting (a required member recurses forever)"
        ));
    }
    let finite: Vec<bool> = (0..nodes.len()).map(|i| fits(i as NodeId)).collect();
    for node in nodes.iter_mut() {
        match node {
            Node::Union(alts) => alts.retain(|&a| finite[a as usize]),
            Node::Object(o) => {
                o.usable = (0..o.names.len())
                    .filter(|&m| finite[o.values[m] as usize])
                    .fold(0u128, |mask, m| mask | (1 << m));
                if o.additional.is_some_and(|a| !finite[a as usize]) {
                    o.additional = None;
                }
            }
            Node::Array(a) => {
                if a.items.is_some_and(|n| !finite[n as usize]) {
                    a.items = None;
                }
                // Items can be written only up to the first that cannot.
                let prefix_ok = a.prefix.iter().take_while(|&&n| finite[n as usize]).count();
                let reachable = if prefix_ok < a.prefix.len() || a.items.is_none() {
                    prefix_ok as u32
                } else {
                    u32::MAX
                };
                a.max = a.max.min(reachable.max(a.min));
            }
            _ => {}
        }
    }
    Ok(depth.into_iter().map(|d| d.min(u32::from(u16::MAX)) as u16).collect())
}

struct Compiler<'a> {
    root: &'a Value,
    nodes: Vec<Node>,
    refs: HashMap<String, NodeId>,
    /// Objects without `additionalProperties` take no undeclared members
    /// (tool arguments: a function accepts only the parameters it declares).
    closed: bool,
}

fn as_count(map: &Map<String, Value>, key: &str) -> Result<Option<u32>, SchemaError> {
    match map.get(key) {
        None => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) => Ok(Some(n.min(u64::from(u32::MAX)) as u32)),
            None => err(format!("'{key}' must be a non-negative integer")),
        },
    }
}

fn as_number(map: &Map<String, Value>, key: &str) -> Result<Option<f64>, SchemaError> {
    match map.get(key) {
        None => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::Bool(_)) => Ok(None),
        Some(_) => err(format!("'{key}' must be a number")),
    }
}

/// The tighter of two lower (or, with `upper`, upper) bounds.
fn tighter(a: Option<Bound>, b: Option<Bound>, upper: bool) -> Option<Bound> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let a_wins = if upper {
                a.value < b.value || (a.value == b.value && a.exclusive)
            } else {
                a.value > b.value || (a.value == b.value && a.exclusive)
            };
            Some(if a_wins { a } else { b })
        }
    }
}

fn bounds(map: &Map<String, Value>) -> Result<(Option<Bound>, Option<Bound>), SchemaError> {
    let draft4_exclusive = |key: &str| map.get(key).and_then(Value::as_bool).unwrap_or(false);
    let lower = as_number(map, "minimum")?.map(|value| Bound {
        value,
        exclusive: draft4_exclusive("exclusiveMinimum"),
    });
    let upper = as_number(map, "maximum")?.map(|value| Bound {
        value,
        exclusive: draft4_exclusive("exclusiveMaximum"),
    });
    let ex_lower = as_number(map, "exclusiveMinimum")?.map(|value| Bound { value, exclusive: true });
    let ex_upper = as_number(map, "exclusiveMaximum")?.map(|value| Bound { value, exclusive: true });
    for b in [lower, upper, ex_lower, ex_upper].into_iter().flatten() {
        if !b.value.is_finite() {
            return err("numeric bounds must be finite");
        }
    }
    Ok((tighter(lower, ex_lower, false), tighter(upper, ex_upper, true)))
}

/// Largest magnitude a bound may have: bounded numbers are written without
/// an exponent in at most 32 characters.
const MAX_BOUND: f64 = 1e30;

/// Whether a number node admits at least one value the decoder can write.
fn number_satisfiable(n: &NumberNode) -> bool {
    if n.lower.is_some_and(|b| b.value > MAX_BOUND) || n.upper.is_some_and(|b| b.value < -MAX_BOUND) {
        return false;
    }
    let (Some(lo), Some(hi)) = (n.lower, n.upper) else {
        return true;
    };
    if n.integer {
        let lo = if lo.exclusive { lo.value.floor() + 1.0 } else { lo.value.ceil() };
        let hi = if hi.exclusive { hi.value.ceil() - 1.0 } else { hi.value.floor() };
        lo <= hi
    } else {
        lo.value < hi.value || (lo.value == hi.value && !lo.exclusive && !hi.exclusive)
    }
}

impl Compiler<'_> {
    fn push(&mut self, node: Node) -> Result<NodeId, SchemaError> {
        if self.nodes.len() >= MAX_NODES {
            return err(format!("the schema is too large (more than {MAX_NODES} nodes)"));
        }
        self.nodes.push(node);
        Ok((self.nodes.len() - 1) as NodeId)
    }

    fn node(&mut self, schema: &Value, depth: usize) -> Result<NodeId, SchemaError> {
        if depth > MAX_SCHEMA_DEPTH {
            return err(format!("the schema nests deeper than {MAX_SCHEMA_DEPTH} levels"));
        }
        let map = match schema {
            Value::Bool(true) => return Ok(ANY),
            Value::Bool(false) => return err("the schema `false` accepts no document"),
            Value::Object(map) => map,
            _ => return err("a schema must be an object or a boolean"),
        };
        if let Some(key) = UNSUPPORTED.iter().find(|k| map.contains_key(**k)) {
            return err(format!("the schema keyword '{key}' is not supported"));
        }
        // Keywords that constrain the value directly (not annotations, not
        // the combinators handled below).
        let constraint_keys = || {
            map.keys()
                .filter(|k| !ANNOTATIONS.contains(&k.as_str()) && !matches!(k.as_str(), "$ref" | "anyOf" | "oneOf" | "allOf"))
                .count()
        };

        if let Some(reference) = map.get("$ref") {
            if constraint_keys() > 0 || map.contains_key("anyOf") || map.contains_key("oneOf") || map.contains_key("allOf") {
                return err("'$ref' alongside other constraints is not supported");
            }
            let reference = reference.as_str().ok_or_else(|| SchemaError("'$ref' must be a string".into()))?;
            return self.reference(reference, depth);
        }
        if let Some(all) = map.get("allOf") {
            let members = all.as_array().ok_or_else(|| SchemaError("'allOf' must be a list".into()))?;
            if members.len() != 1 || constraint_keys() > 0 || map.contains_key("anyOf") || map.contains_key("oneOf") {
                return err("'allOf' is supported only with a single member and no sibling constraints");
            }
            return self.node(&members[0], depth + 1);
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(alts) = map.get(key) {
                if constraint_keys() > 0 {
                    return err(format!("'{key}' alongside other constraints is not supported"));
                }
                let alts = alts.as_array().ok_or_else(|| SchemaError(format!("'{key}' must be a list")))?;
                if alts.is_empty() {
                    return err(format!("'{key}' must not be empty"));
                }
                let mut ids = Vec::with_capacity(alts.len());
                for alt in alts {
                    ids.push(self.node(alt, depth + 1)?);
                }
                return self.union(ids);
            }
        }

        let nullable = map.get("nullable").and_then(Value::as_bool).unwrap_or(false);
        let types: Vec<&str> = match map.get("type") {
            None => Vec::new(),
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(ts)) => ts
                .iter()
                .map(|t| t.as_str().ok_or_else(|| SchemaError("'type' entries must be strings".into())))
                .collect::<Result<_, _>>()?,
            Some(_) => return err("'type' must be a string or a list of strings"),
        };
        for t in &types {
            if !matches!(*t, "object" | "array" | "string" | "number" | "integer" | "boolean" | "null") {
                return err(format!("unknown type '{t}'"));
            }
        }

        if let Some(values) = map.get("enum") {
            let values = values.as_array().ok_or_else(|| SchemaError("'enum' must be a list".into()))?;
            return self.literals(values.clone(), &types, nullable);
        }
        if let Some(value) = map.get("const") {
            return self.literals(vec![value.clone()], &types, nullable);
        }

        // Without `type`, the type-specific keywords apply to their own kind
        // and every other kind is unconstrained.
        let kinds: Vec<&str> = if types.is_empty() {
            if constraint_keys() == usize::from(map.contains_key("nullable")) {
                return Ok(ANY);
            }
            vec!["object", "array", "string", "number", "boolean", "null"]
        } else {
            types
        };
        let mut ids = Vec::new();
        for kind in kinds {
            ids.push(self.typed(kind, map, depth)?);
        }
        if nullable && !ids.contains(&NULL) {
            ids.push(NULL);
        }
        self.union(ids)
    }

    fn union(&mut self, mut ids: Vec<NodeId>) -> Result<NodeId, SchemaError> {
        ids.dedup();
        if ids.len() == 1 {
            return Ok(ids[0]);
        }
        self.push(Node::Union(ids))
    }

    fn literals(&mut self, values: Vec<Value>, types: &[&str], nullable: bool) -> Result<NodeId, SchemaError> {
        let fits = |v: &Value| {
            types.is_empty()
                || (nullable && v.is_null())
                || types.iter().any(|t| match *t {
                    "object" => v.is_object(),
                    "array" => v.is_array(),
                    "string" => v.is_string(),
                    "number" => v.is_number(),
                    "integer" => v.as_i64().is_some() || v.as_u64().is_some(),
                    "boolean" => v.is_boolean(),
                    "null" => v.is_null(),
                    _ => false,
                })
        };
        let mut texts: Vec<Vec<u8>> = values
            .iter()
            .filter(|v| fits(v))
            .map(|v| serde_json::to_vec(v).expect("a JSON value serializes"))
            .collect();
        if texts.is_empty() {
            return err("'enum' / 'const' leaves no value of the allowed types");
        }
        texts.sort();
        texts.dedup();
        self.push(Node::Literals(texts))
    }

    fn typed(&mut self, kind: &str, map: &Map<String, Value>, depth: usize) -> Result<NodeId, SchemaError> {
        match kind {
            "boolean" => Ok(BOOLEAN),
            "null" => Ok(NULL),
            "string" => {
                let min = as_count(map, "minLength")?.unwrap_or(0);
                let max = as_count(map, "maxLength")?.unwrap_or(u32::MAX);
                if min > max {
                    return err("minLength exceeds maxLength");
                }
                if (min, max) == (0, u32::MAX) {
                    return Ok(ANY_STRING);
                }
                self.push(Node::String { min, max })
            }
            "number" | "integer" => {
                let (lower, upper) = bounds(map)?;
                let node = NumberNode {
                    integer: kind == "integer",
                    lower,
                    upper,
                };
                if !number_satisfiable(&node) {
                    return err("the numeric bounds admit no value");
                }
                if node == (NumberNode { integer: false, lower: None, upper: None }) {
                    return Ok(ANY_NUMBER);
                }
                self.push(Node::Number(node))
            }
            "array" => self.array(map, depth),
            "object" => self.object(map, depth),
            _ => unreachable!("types are checked above"),
        }
    }

    fn array(&mut self, map: &Map<String, Value>, depth: usize) -> Result<NodeId, SchemaError> {
        let mut prefix = Vec::new();
        if let Some(items) = map.get("prefixItems") {
            let items = items.as_array().ok_or_else(|| SchemaError("'prefixItems' must be a list".into()))?;
            for item in items {
                prefix.push(self.node(item, depth + 1)?);
            }
        }
        let items = match map.get("items") {
            None => Some(ANY),
            Some(Value::Bool(false)) => None,
            // Draft 4-7 tuple form.
            Some(Value::Array(tuple)) => {
                if !prefix.is_empty() {
                    return err("'items' as a list cannot be combined with 'prefixItems'");
                }
                for item in tuple {
                    prefix.push(self.node(item, depth + 1)?);
                }
                Some(ANY)
            }
            Some(schema) => Some(self.node(schema, depth + 1)?),
        };
        let min = as_count(map, "minItems")?.unwrap_or(0);
        let mut max = as_count(map, "maxItems")?.unwrap_or(u32::MAX);
        if items.is_none() {
            max = max.min(prefix.len() as u32);
        }
        if min > max {
            return err("minItems exceeds what the array schema allows");
        }
        if prefix.is_empty() && items == Some(ANY) && (min, max) == (0, u32::MAX) {
            return Ok(ANY_ARRAY);
        }
        self.push(Node::Array(ArrayNode { prefix, items, min, max }))
    }

    fn object(&mut self, map: &Map<String, Value>, depth: usize) -> Result<NodeId, SchemaError> {
        let additional = match map.get("additionalProperties") {
            None if self.closed => None,
            None | Some(Value::Bool(true)) => Some(ANY),
            Some(Value::Bool(false)) => None,
            Some(schema) => Some(self.node(schema, depth + 1)?),
        };
        let mut members: Vec<(Vec<u8>, NodeId)> = Vec::new();
        if let Some(props) = map.get("properties") {
            let props = props.as_object().ok_or_else(|| SchemaError("'properties' must be an object".into()))?;
            for (name, schema) in props {
                members.push((name.as_bytes().to_vec(), self.node(schema, depth + 1)?));
            }
        }
        let required: Vec<&str> = match map.get("required") {
            None => Vec::new(),
            Some(Value::Array(names)) => names
                .iter()
                .map(|n| n.as_str().ok_or_else(|| SchemaError("'required' entries must be strings".into())))
                .collect::<Result<_, _>>()?,
            Some(_) => return err("'required' must be a list of strings"),
        };
        for name in &required {
            if !members.iter().any(|(n, _)| n == name.as_bytes()) {
                match additional {
                    // A required member covered only by additionalProperties.
                    Some(schema) => members.push((name.as_bytes().to_vec(), schema)),
                    None => return err(format!("required member '{name}' is not allowed by the schema")),
                }
            }
        }
        if members.len() > MAX_PROPERTIES {
            return err(format!("an object schema may declare at most {MAX_PROPERTIES} members"));
        }
        members.sort_by(|a, b| a.0.cmp(&b.0));
        let mut mask = 0u128;
        for name in &required {
            let i = members.iter().position(|(n, _)| n == name.as_bytes()).expect("added above");
            mask |= 1 << i;
        }
        let min = as_count(map, "minProperties")?.unwrap_or(0);
        let max = as_count(map, "maxProperties")?.unwrap_or(u32::MAX);
        let required_count = mask.count_ones();
        let possible = if additional.is_some() { u32::MAX } else { members.len() as u32 };
        if required_count > max || min > max || min > possible {
            return err("the object schema's member counts admit no object");
        }
        let (names, values) = members.into_iter().unzip();
        self.push(Node::Object(ObjectNode {
            names,
            values,
            required: mask,
            usable: u128::MAX,
            additional,
            min,
            max,
        }))
    }

    fn reference(&mut self, reference: &str, depth: usize) -> Result<NodeId, SchemaError> {
        if let Some(&id) = self.refs.get(reference) {
            return Ok(id);
        }
        let pointer = reference
            .strip_prefix('#')
            .ok_or_else(|| SchemaError(format!("only local references (#...) are supported, not '{reference}'")))?;
        let target = if pointer.is_empty() {
            self.root
        } else {
            self.root
                .pointer(pointer)
                .ok_or_else(|| SchemaError(format!("'$ref' target '{reference}' does not exist")))?
        };
        // The alias exists before its target compiles, so recursive
        // references resolve to it.
        let alias = self.push(Node::Alias(ANY))?;
        self.refs.insert(reference.to_string(), alias);
        let target = self.node(target, depth + 1)?;
        if target == alias {
            return err(format!("'$ref' '{reference}' refers only to itself"));
        }
        self.nodes[alias as usize] = Node::Alias(target);
        Ok(alias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unsupported_keywords_are_refused_by_name() {
        let e = Schema::compile(&json!({"type": "string", "pattern": "^a"})).unwrap_err();
        assert!(e.to_string().contains("'pattern'"), "{e}");
        let e = Schema::compile(&json!({"type": "object", "properties": {"x": {"not": {}}}})).unwrap_err();
        assert!(e.to_string().contains("'not'"), "{e}");
    }

    #[test]
    fn unsatisfiable_schemas_are_refused() {
        assert!(Schema::compile(&json!(false)).is_err());
        assert!(Schema::compile(&json!({"type": "integer", "minimum": 2, "maximum": 1})).is_err());
        assert!(Schema::compile(&json!({"type": "integer", "exclusiveMinimum": 1, "exclusiveMaximum": 2})).is_err());
        assert!(
            Schema::compile(&json!({"type": "object", "required": ["a"], "additionalProperties": false})).is_err()
        );
        assert!(Schema::compile(&json!({"type": "string", "enum": [1, 2]})).is_err());
    }

    #[test]
    fn annotations_and_trivial_schemas_compile_to_builtins() {
        assert_eq!(Schema::compile(&json!({})).unwrap().root, ANY);
        assert_eq!(Schema::compile(&json!({"title": "t", "format": "date-time"})).unwrap().root, ANY);
        assert_eq!(Schema::compile(&json!({"type": "string", "description": "d"})).unwrap().root, ANY_STRING);
    }

    #[test]
    fn recursive_references_resolve() {
        let schema = json!({
            "$defs": {"node": {"type": "object", "properties": {"next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}}}},
            "$ref": "#/$defs/node",
        });
        let s = Schema::compile(&schema).unwrap();
        assert!(matches!(s.resolve(s.root), Node::Object(_)));
        assert!(Schema::compile(&json!({"$ref": "#/$defs/missing"})).is_err());
    }

    #[test]
    fn schemas_without_finite_documents_are_refused_and_dead_branches_pruned() {
        let forever = json!({"type": "object", "properties": {"a": {"$ref": "#"}}, "required": ["a"]});
        let e = Schema::compile(&forever).unwrap_err();
        assert!(e.to_string().contains("recurses forever"), "{e}");
        // Optional recursion ends at `{}`: fine.
        assert!(Schema::compile(&json!({"type": "object", "properties": {"a": {"$ref": "#"}}})).is_ok());
        // A union branch that can never finish is dropped.
        let s = Schema::compile(&json!({
            "anyOf": [{"type": "null"}, {"$ref": "#/$defs/l"}],
            "$defs": {"l": {"type": "object", "properties": {"x": {"$ref": "#/$defs/l"}}, "required": ["x"]}},
        }))
        .unwrap();
        assert_eq!(s.resolve(s.root), &Node::Union(vec![NULL]));
        // Bounds no 32-character number can reach are refused.
        assert!(Schema::compile(&json!({"type": "integer", "minimum": 1e40})).is_err());
    }

    #[test]
    fn required_members_are_sorted_with_their_bits() {
        let s = Schema::compile(&json!({
            "type": "object",
            "properties": {"b": {"type": "string"}, "a": {"type": "integer"}},
            "required": ["b"],
            "additionalProperties": false,
        }))
        .unwrap();
        let Node::Object(o) = s.resolve(s.root) else { panic!("an object") };
        assert_eq!(o.names, vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(o.required, 0b10);
        assert_eq!(o.additional, None);
    }
}
