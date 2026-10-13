//! A Llama-family decoder (Llama 2/3, Mistral, Qwen 2 and others with the
//! same layout) loaded from a Hugging Face checkpoint directory: dense or
//! packed weights, grouped-query attention, RMS norm, rotary embeddings and
//! a SwiGLU MLP.
//!
//! [`Llama::forward_batch`] advances several independent sequences by one
//! token each in one step: every projection and MLP runs once on the rows of
//! all sequences, so each weight matrix is read once per step for the whole
//! batch, which is where decoding spends its time. Only rotary embedding and
//! attention run per sequence, each against its own cache length.
//!
//! Callers: applications through [`LanguageModel`] (`Session`), and the
//! serving engine in `cortex-serve` (its `ServeModel` impl for [`Llama`]).
//! Calls: `checkpoint.rs` (weights), `quantized.rs` (packed projections),
//! `fused.rs` (`rms_norm`, `rope`), `cache.rs` (`KvCache`,
//! `cache_attention`). Sibling: `cortex-nn`'s attention modules
//! (`modules/attention`), the training-oriented counterpart.

use std::path::Path;

use cortex_core::tensor::activation::silu;
use cortex_core::tensor::{Device, Int, Tensor, TensorData};

use crate::cache::{KvCache, cache_attention};
use crate::checkpoint::{Checkpoint, CheckpointError};
use crate::fused::{rms_norm, rope, rope_inv_freq};
use crate::generate::LanguageModel;
use crate::quantized::QuantizedMatrix;

/// The fields of a Hugging Face `config.json` this model reads.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Width of the MLP's hidden layer.
    pub intermediate_size: usize,
    /// Decoder layers.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads (grouped-query attention); equal to the query heads
    /// when the config omits it.
    pub num_key_value_heads: usize,
    /// Width of one head; `hidden_size / num_attention_heads` when omitted.
    pub head_dim: usize,
    /// Token ids.
    pub vocab_size: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Rotary embedding base.
    pub rope_theta: f32,
    /// Whether the output head reuses the token embedding.
    pub tie_word_embeddings: bool,
}

/// Why a model could not be loaded.
#[derive(Debug)]
pub enum ModelError {
    /// The config is missing, unreadable or inconsistent.
    Config(String),
    /// A weight is missing or malformed.
    Checkpoint(CheckpointError),
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Config(why) => write!(f, "model config: {why}"),
            Self::Checkpoint(e) => write!(f, "model weights: {e:?}"),
        }
    }
}

impl From<CheckpointError> for ModelError {
    fn from(e: CheckpointError) -> Self {
        Self::Checkpoint(e)
    }
}

impl LlamaConfig {
    /// Parses a `config.json`, checking the shapes it implies.
    pub fn from_json(text: &str) -> Result<Self, ModelError> {
        let json: serde_json::Value = serde_json::from_str(text).map_err(|e| ModelError::Config(e.to_string()))?;
        let size = |key: &str| -> Result<usize, ModelError> {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| ModelError::Config(format!("`{key}` is missing or not a count")))
        };
        let float = |key: &str, default: f64| json.get(key).and_then(serde_json::Value::as_f64).unwrap_or(default) as f32;
        let hidden_size = size("hidden_size")?;
        let num_attention_heads = size("num_attention_heads")?;
        let num_key_value_heads = size("num_key_value_heads").unwrap_or(num_attention_heads);
        if num_attention_heads == 0 || num_key_value_heads == 0 || num_attention_heads % num_key_value_heads != 0 {
            return Err(ModelError::Config(format!(
                "{num_attention_heads} query heads cannot share {num_key_value_heads} key/value heads"
            )));
        }
        let head_dim = match json.get("head_dim").and_then(serde_json::Value::as_u64) {
            Some(d) => d as usize,
            None if hidden_size % num_attention_heads == 0 => hidden_size / num_attention_heads,
            None => return Err(ModelError::Config("`head_dim` is missing and heads do not divide the width".into())),
        };
        if head_dim % 2 != 0 {
            return Err(ModelError::Config(format!("rotary embedding needs an even head width, got {head_dim}")));
        }
        Ok(Self {
            hidden_size,
            intermediate_size: size("intermediate_size")?,
            num_hidden_layers: size("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size: size("vocab_size")?,
            rms_norm_eps: float("rms_norm_eps", 1e-6),
            rope_theta: float("rope_theta", 10_000.0),
            tie_word_embeddings: json
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
    }
}

/// A projection `x[m, in] → [m, out]`, dense or packed. Both handles are
/// boxed: they differ several-fold in size, and a model holds seven per
/// layer.
#[derive(Clone, Debug)]
enum Linear {
    /// Stored as the checkpoint has it, `[out, in]`.
    Dense(Box<Tensor<2>>),
    /// Packed `[in, out]` (see `checkpoint::quantize_checkpoint`).
    Packed(Box<QuantizedMatrix>),
}

impl Linear {
    fn load(ckpt: &Checkpoint, packed: bool, name: &str, shape: [usize; 2], device: &Device) -> Result<Self, ModelError> {
        let [out, input] = shape;
        let linear = if packed {
            let w = ckpt.quantized_matrix(name, device)?;
            if (w.k(), w.n()) != (input, out) {
                return Err(ModelError::Config(format!(
                    "{name} is packed as [{}, {}], expected [{input}, {out}]",
                    w.k(),
                    w.n()
                )));
            }
            Self::Packed(Box::new(w))
        } else {
            let w = ckpt.tensor::<2>(name, device)?;
            if w.dims() != [out, input] {
                return Err(ModelError::Config(format!("{name} is {:?}, expected [{out}, {input}]", w.dims())));
            }
            Self::Dense(Box::new(w))
        };
        Ok(linear)
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        match self {
            Self::Dense(w) => x.matmul(w.clone().transpose()),
            Self::Packed(w) => w.forward(x),
        }
    }
}

#[derive(Clone, Debug)]
struct Layer {
    input_norm: Tensor<1>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    post_norm: Tensor<1>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// A Llama-family decoder on one device.
#[derive(Clone, Debug)]
pub struct Llama {
    config: LlamaConfig,
    /// `[vocab, hidden]`.
    embed: Tensor<2>,
    layers: Vec<Layer>,
    norm: Tensor<1>,
    head: Linear,
    inv_freq: Tensor<1>,
    device: Device,
}

/// The caches of one sequence: one [`KvCache`] per layer.
pub type LlamaState = Vec<KvCache>;

impl Llama {
    /// Loads `dir`: its `config.json` and its safetensors (one file, a
    /// directory of them, or a sharded index). Weights packed by
    /// `quantize_checkpoint` load packed; everything else as stored.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| ModelError::Config(format!("{}: {e}", config_path.display())))?;
        let config = LlamaConfig::from_json(&text)?;
        let ckpt = Checkpoint::open(dir)?;
        Self::from_checkpoint(config, &ckpt, device)
    }

    /// Builds the model from an open checkpoint.
    pub fn from_checkpoint(config: LlamaConfig, ckpt: &Checkpoint, device: &Device) -> Result<Self, ModelError> {
        let packed = ckpt.quant_formats()?;
        let c = &config;
        let (h, hd) = (c.hidden_size, c.head_dim);
        let linear = |name: &str, shape: [usize; 2]| Linear::load(ckpt, packed.contains_key(name), name, shape, device);
        let vector = |name: &str, len: usize| -> Result<Tensor<1>, ModelError> {
            let t = ckpt.tensor::<1>(name, device)?;
            if t.dims() != [len] {
                return Err(ModelError::Config(format!("{name} is {:?}, expected [{len}]", t.dims())));
            }
            Ok(t)
        };

        let embed_name = "model.embed_tokens.weight";
        let embed = match linear(embed_name, [c.vocab_size, h])? {
            Linear::Dense(w) => *w,
            // Packed as `[hidden, vocab]`; a lookup needs rows per token.
            Linear::Packed(w) => w.dequantize().transpose(),
        };
        let head = if ckpt.contains("lm_head.weight") || packed.contains_key("lm_head.weight") {
            linear("lm_head.weight", [c.vocab_size, h])?
        } else if c.tie_word_embeddings {
            linear(embed_name, [c.vocab_size, h])?
        } else {
            return Err(ModelError::Config("no `lm_head.weight` and embeddings are not tied".into()));
        };

        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: vector(&format!("{p}.input_layernorm.weight"), h)?,
                q: linear(&format!("{p}.self_attn.q_proj.weight"), [c.num_attention_heads * hd, h])?,
                k: linear(&format!("{p}.self_attn.k_proj.weight"), [c.num_key_value_heads * hd, h])?,
                v: linear(&format!("{p}.self_attn.v_proj.weight"), [c.num_key_value_heads * hd, h])?,
                o: linear(&format!("{p}.self_attn.o_proj.weight"), [h, c.num_attention_heads * hd])?,
                post_norm: vector(&format!("{p}.post_attention_layernorm.weight"), h)?,
                gate: linear(&format!("{p}.mlp.gate_proj.weight"), [c.intermediate_size, h])?,
                up: linear(&format!("{p}.mlp.up_proj.weight"), [c.intermediate_size, h])?,
                down: linear(&format!("{p}.mlp.down_proj.weight"), [h, c.intermediate_size])?,
            });
        }
        Ok(Self {
            embed,
            layers,
            norm: vector("model.norm.weight", h)?,
            head,
            inv_freq: rope_inv_freq(hd, c.rope_theta, device),
            device: device.clone(),
            config,
        })
    }

    /// The model's config.
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A state holding no positions.
    pub fn new_state(&self) -> LlamaState {
        (0..self.layers.len()).map(|_| KvCache::default()).collect()
    }

    /// Bytes `state` holds in its caches (keys and values of every layer).
    pub fn state_bytes(&self, state: &LlamaState) -> usize {
        let per_position = 2 * self.config.num_key_value_heads * self.config.head_dim * self.embed.dtype().size();
        state.first().map_or(0, KvCache::offset) * per_position * state.len()
    }

    fn embed(&self, tokens: &[u32]) -> Tensor<2> {
        assert!(
            tokens.iter().all(|&t| (t as usize) < self.config.vocab_size),
            "token id out of the vocabulary of {}",
            self.config.vocab_size
        );
        let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
        let ids = Tensor::<1, Int>::from_data(TensorData::new(ids, [tokens.len()]), &self.device);
        self.embed.clone().select(0, ids)
    }

    /// Feeds `tokens` after the positions `state` holds and returns the
    /// logits predicting the token after each, `[tokens.len(), vocab]`.
    pub fn forward(&self, state: &mut LlamaState, tokens: &[u32]) -> Tensor<2> {
        assert!(!tokens.is_empty(), "nothing to feed");
        let mut states = [state];
        self.run(&mut states, &[tokens])
    }

    /// One token for each of several sequences in one step: `tokens[i]`
    /// follows `states[i]`. Returns `[states.len(), vocab]`, row `i` the
    /// logits of sequence `i` — the same as [`forward`](Self::forward) on
    /// each alone, with the weights read once for all of them.
    pub fn forward_batch(&self, states: &mut [&mut LlamaState], tokens: &[u32]) -> Tensor<2> {
        assert_eq!(states.len(), tokens.len(), "one token per sequence");
        assert!(!tokens.is_empty(), "nothing to feed");
        let per_sequence: Vec<&[u32]> = tokens.chunks(1).collect();
        self.run(states, &per_sequence)
    }

    /// Runs `chunks[i]` after `states[i]` for every `i`: the rows of all
    /// sequences go through each projection together.
    fn run(&self, states: &mut [&mut LlamaState], chunks: &[&[u32]]) -> Tensor<2> {
        let c = &self.config;
        let (hq, hkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let all: Vec<u32> = chunks.iter().flat_map(|t| t.iter().copied()).collect();
        let mut x = self.embed(&all);

        for (li, layer) in self.layers.iter().enumerate() {
            let h = rms_norm(x.clone(), layer.input_norm.clone(), c.rms_norm_eps, false);
            let (q, k, v) = (layer.q.forward(h.clone()), layer.k.forward(h.clone()), layer.v.forward(h));

            let mut outs = Vec::with_capacity(chunks.len());
            let mut row = 0;
            for (state, chunk) in states.iter_mut().zip(chunks) {
                let n = chunk.len();
                let heads = |t: &Tensor<2>, count: usize| t.clone().narrow(0, row, n).reshape([1, n, count, hd]).swap_dims(1, 2);
                let cache = &mut state[li];
                let offset = cache.offset();
                let qs = rope(heads(&q, hq), self.inv_freq.clone(), offset, false);
                let ks = rope(heads(&k, hkv), self.inv_freq.clone(), offset, false);
                let (keys, values) = cache.update(ks, heads(&v, hkv));
                let attn = cache_attention(qs, keys, values, None);
                outs.push(attn.swap_dims(1, 2).reshape([n, hq * hd]));
                row += n;
            }
            x = x + layer.o.forward(Tensor::cat(outs, 0));

            let h = rms_norm(x.clone(), layer.post_norm.clone(), c.rms_norm_eps, false);
            let mlp = silu(layer.gate.forward(h.clone())) * layer.up.forward(h);
            x = x + layer.down.forward(mlp);
        }
        self.head.forward(rms_norm(x, self.norm.clone(), c.rms_norm_eps, false))
    }

    /// A single-sequence [`LanguageModel`] over this model, for
    /// [`generate`](crate::generate) and speculative decoding.
    pub fn session(&self) -> Session<'_> {
        Session {
            model: self,
            state: self.new_state(),
        }
    }
}

/// One sequence being generated by a [`Llama`].
pub struct Session<'a> {
    model: &'a Llama,
    state: LlamaState,
}

impl Session<'_> {
    /// The sequence's caches.
    pub fn state(&self) -> &LlamaState {
        &self.state
    }
}

impl LanguageModel for Session<'_> {
    fn forward(&mut self, tokens: &[u32]) -> Tensor<2> {
        self.model.forward(&mut self.state, tokens)
    }

    fn trim(&mut self, n: usize) {
        for cache in &mut self.state {
            cache.trim(n);
        }
    }
}
