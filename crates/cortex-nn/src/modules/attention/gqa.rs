use cortex_core as cortex;

use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use cortex::config::Config;
use cortex::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay};
use cortex::tensor::{Bool, Device, Tensor};

use cortex::tensor::activation::{quiet_softmax, softmax};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Configuration to create a [Group Query Attention](GroupQueryAttention) layer
/// using the [init function](GroupQueryAttentionConfig::init).
///
/// Group Query Attention (GQA) generalizes Multi-Head Attention by allowing multiple query heads
/// to share the same key and value heads. This reduces memory and compute for the KV cache
/// while maintaining model quality.
///
/// Special cases:
/// - `n_kv_heads == n_query_heads`: equivalent to standard Multi-Head Attention (MHA)
/// - `n_kv_heads == 1`: equivalent to Multi-Query Attention (MQA)
///
/// Reference: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
#[derive(Config, Debug)]
pub struct GroupQueryAttentionConfig {
    /// The model dimension.
    pub d_model: usize,
    /// The number of query heads.
    pub n_query_heads: usize,
    /// The number of key/value heads. Must evenly divide `n_query_heads`.
    pub n_kv_heads: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// The minimum value a float can take for masking. Default: -1.0e4
    #[config(default = -1.0e4)]
    pub min_float: f64,
    /// Use "quiet softmax" instead of regular softmax. Default: false
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Group Query Attention module.
///
/// Generalizes Multi-Head Attention by allowing multiple query heads to share key/value heads.
/// This reduces KV cache memory by a factor of `n_query_heads / n_kv_heads` while maintaining
/// model quality close to full MHA.
///
/// # Params
///
/// - `query`: [`Linear`] layer projecting to `n_query_heads * head_dim`.
/// - `key`: [`Linear`] layer projecting to `n_kv_heads * head_dim`.
/// - `value`: [`Linear`] layer projecting to `n_kv_heads * head_dim`.
/// - `output`: [`Linear`] layer projecting back to `d_model`.
///
/// Should be created with [GroupQueryAttentionConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct GroupQueryAttention {
    /// Linear layer for query projection.
    pub query: Linear,
    /// Linear layer for key projection.
    pub key: Linear,
    /// Linear layer for value projection.
    pub value: Linear,
    /// Linear layer for output projection.
    pub output: Linear,
    /// Dropout layer.
    pub dropout: Dropout,
    /// Model dimension.
    pub d_model: usize,
    /// Number of query heads.
    pub n_query_heads: usize,
    /// Number of key/value heads.
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Number of query heads per KV head.
    pub n_groups: usize,
    /// Minimum float value for masking.
    pub min_float: f64,
    /// Whether to use quiet softmax.
    pub quiet_softmax: bool,
}

impl ModuleDisplay for GroupQueryAttention {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_query_heads", &self.n_query_heads)
            .add("n_kv_heads", &self.n_kv_heads)
            .add("head_dim", &self.head_dim)
            .add("dropout", &self.dropout.prob)
            .optional()
    }
}

/// [Group Query Attention](GroupQueryAttention) forward pass input argument.
#[derive(Debug, Clone)]
pub struct GqaInput {
    /// Query tensor of shape `[batch_size, seq_length_1, d_model]`.
    query: Tensor<3>,
    /// Key tensor of shape `[batch_size, seq_length_2, d_model]`.
    key: Tensor<3>,
    /// Value tensor of shape `[batch_size, seq_length_2, d_model]`.
    value: Tensor<3>,
    mask_pad: Option<Tensor<2, Bool>>,
    mask_attn: Option<Tensor<3, Bool>>,
}

impl GqaInput {
    /// Create a self-attention input (query = key = value).
    pub fn self_attn(tensor: Tensor<3>) -> Self {
        Self {
            query: tensor.clone(),
            key: tensor.clone(),
            value: tensor,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Create a cross-attention input.
    pub fn new(query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> Self {
        Self {
            query,
            key,
            value,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Register a padding mask.
    pub fn mask_pad(mut self, mask_pad: Tensor<2, Bool>) -> Self {
        self.mask_pad = Some(mask_pad);
        self
    }

    /// Register an attention mask.
    pub fn mask_attn(mut self, mask_attn: Tensor<3, Bool>) -> Self {
        self.mask_attn = Some(mask_attn);
        self
    }
}

/// [Group Query Attention](GroupQueryAttention) outputs.
#[derive(Debug, Clone)]
pub struct GqaOutput {
    /// The attention weights `[batch_size, n_query_heads, seq_length_1, seq_length_2]`.
    pub weights: Tensor<4>,
    /// The context tensor `[batch_size, seq_length_1, d_model]`.
    pub context: Tensor<3>,
}

impl GroupQueryAttentionConfig {
    /// Initialize a new [Group Query Attention](GroupQueryAttention) module.
    ///
    /// # Panics
    ///
    /// Panics if `n_query_heads` is not divisible by `n_kv_heads`.
    /// Panics if `d_model` is not divisible by `n_query_heads`.
    pub fn init(&self, device: &Device) -> GroupQueryAttention {
        assert!(
            self.n_query_heads % self.n_kv_heads == 0,
            "n_query_heads ({}) must be divisible by n_kv_heads ({})",
            self.n_query_heads,
            self.n_kv_heads
        );
        assert!(
            self.d_model % self.n_query_heads == 0,
            "d_model ({}) must be divisible by n_query_heads ({})",
            self.d_model,
            self.n_query_heads
        );

        let head_dim = self.d_model / self.n_query_heads;
        let n_groups = self.n_query_heads / self.n_kv_heads;

        let query = LinearConfig::new(self.d_model, self.n_query_heads * head_dim)
            .with_initializer(self.initializer.clone())
            .init(device);
        let key = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_initializer(self.initializer.clone())
            .init(device);
        let value = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_initializer(self.initializer.clone())
            .init(device);
        let output = LinearConfig::new(self.d_model, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);

        GroupQueryAttention {
            query,
            key,
            value,
            output,
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_query_heads: self.n_query_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
            n_groups,
            min_float: self.min_float,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

impl GroupQueryAttention {
    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward(&self, input: GqaInput) -> GqaOutput {
        let [batch_size, seq_length_1, _d_model] = input.query.dims();
        let seq_length_2 = input.key.dims()[1];

        // Project Q, K, V
        // Q: [batch, seq1, n_query_heads * head_dim] -> [batch, n_query_heads, seq1, head_dim]
        let q = self.query.forward(input.query);
        let q = q
            .reshape([batch_size, seq_length_1, self.n_query_heads, self.head_dim])
            .swap_dims(1, 2);

        // K: [batch, seq2, n_kv_heads * head_dim] -> [batch, n_kv_heads, seq2, head_dim]
        let k = self.key.forward(input.key);
        let k = k
            .reshape([batch_size, seq_length_2, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // V: [batch, seq2, n_kv_heads * head_dim] -> [batch, n_kv_heads, seq2, head_dim]
        let v = self.value.forward(input.value);
        let v = v
            .reshape([batch_size, seq_length_2, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Expand KV heads to match query heads by repeating each KV head n_groups times
        // K: [batch, n_kv_heads, seq2, head_dim] -> [batch, n_query_heads, seq2, head_dim]
        let k = if self.n_groups > 1 {
            // [batch, n_kv_heads, 1, seq2, head_dim] -> repeat -> [batch, n_kv_heads, n_groups, seq2, head_dim]
            // -> reshape [batch, n_query_heads, seq2, head_dim]
            k.unsqueeze_dim::<5>(2)
                .repeat_dim(2, self.n_groups)
                .reshape([batch_size, self.n_query_heads, seq_length_2, self.head_dim])
        } else {
            k
        };

        let v = if self.n_groups > 1 {
            v.unsqueeze_dim::<5>(2)
                .repeat_dim(2, self.n_groups)
                .reshape([batch_size, self.n_query_heads, seq_length_2, self.head_dim])
        } else {
            v
        };

        // Scaled dot-product attention
        // [batch, n_query_heads, seq1, head_dim] @ [batch, n_query_heads, head_dim, seq2]
        // -> [batch, n_query_heads, seq1, seq2]
        let attn_scores = q
            .matmul(k.transpose())
            .div_scalar((self.head_dim as f32).sqrt());
        let attn_scores = self.dropout.forward(attn_scores);

        // Apply masks
        let weights = self.apply_masks(attn_scores, input.mask_pad, input.mask_attn);

        // [batch, n_query_heads, seq1, seq2] @ [batch, n_query_heads, seq2, head_dim]
        // -> [batch, n_query_heads, seq1, head_dim]
        let context = weights.clone().matmul(v);

        // Reshape back: [batch, n_query_heads, seq1, head_dim] -> [batch, seq1, d_model]
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, self.d_model]);
        let context = self.output.forward(context);

        GqaOutput { weights, context }
    }

    fn apply_masks(
        &self,
        mut attn_scores: Tensor<4>,
        mask_pad: Option<Tensor<2, Bool>>,
        mask_attn: Option<Tensor<3, Bool>>,
    ) -> Tensor<4> {
        if let Some(mask_pad) = mask_pad {
            let [batch_size, seq_length] = mask_pad.dims();
            attn_scores = attn_scores.mask_fill(
                mask_pad.reshape([batch_size, 1, 1, seq_length]),
                self.min_float,
            );
        }

        if let Some(mask_attn) = mask_attn {
            let [batch_size, seq_length_1, seq_length_2] = mask_attn.dims();
            attn_scores = attn_scores.mask_fill(
                mask_attn.reshape([batch_size, 1, seq_length_1, seq_length_2]),
                self.min_float,
            );
        }

        if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex::tensor::Distribution;

    #[test]
    fn test_gqa_output_shape() {
        let device = Default::default();
        let gqa = GroupQueryAttentionConfig::new(64, 8, 2).init(&device);

        let input = Tensor::random([2, 10, 64], Distribution::Default, &device);
        let output = gqa.forward(GqaInput::self_attn(input));

        assert_eq!(output.context.dims(), [2, 10, 64]);
        assert_eq!(output.weights.dims(), [2, 8, 10, 10]);
    }

    #[test]
    fn test_gqa_mqa_output_shape() {
        // n_kv_heads=1 is Multi-Query Attention
        let device = Default::default();
        let gqa = GroupQueryAttentionConfig::new(64, 8, 1).init(&device);

        let input = Tensor::random([2, 10, 64], Distribution::Default, &device);
        let output = gqa.forward(GqaInput::self_attn(input));

        assert_eq!(output.context.dims(), [2, 10, 64]);
        assert_eq!(output.weights.dims(), [2, 8, 10, 10]);
    }

    #[test]
    fn test_gqa_equals_mha_when_same_heads() {
        // When n_kv_heads == n_query_heads, GQA should behave like MHA
        let device = Default::default();
        let gqa = GroupQueryAttentionConfig::new(64, 8, 8).init(&device);

        let input = Tensor::random([2, 10, 64], Distribution::Default, &device);
        let output = gqa.forward(GqaInput::self_attn(input));

        // Just verify shapes are correct
        assert_eq!(output.context.dims(), [2, 10, 64]);
        assert_eq!(output.weights.dims(), [2, 8, 10, 10]);
    }

    #[test]
    fn test_gqa_cross_attention() {
        let device = Default::default();
        let gqa = GroupQueryAttentionConfig::new(64, 8, 4).init(&device);

        let query = Tensor::random([2, 5, 64], Distribution::Default, &device);
        let key = Tensor::random([2, 10, 64], Distribution::Default, &device);
        let value = Tensor::random([2, 10, 64], Distribution::Default, &device);

        let output = gqa.forward(GqaInput::new(query, key, value));

        assert_eq!(output.context.dims(), [2, 5, 64]);
        assert_eq!(output.weights.dims(), [2, 8, 5, 10]);
    }

    #[test]
    #[should_panic]
    fn test_gqa_invalid_head_ratio() {
        let device = Default::default();
        // 8 query heads not divisible by 3 kv heads
        let _gqa = GroupQueryAttentionConfig::new(64, 8, 3).init(&device);
    }

    #[test]
    fn display() {
        let device = Default::default();
        let gqa = GroupQueryAttentionConfig::new(64, 8, 2).init(&device);

        let s = alloc::format!("{gqa}");
        assert!(s.contains("d_model: 64"));
        assert!(s.contains("n_query_heads: 8"));
        assert!(s.contains("n_kv_heads: 2"));
    }
}
