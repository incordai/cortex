//! QLoRA over packed weights: a frozen quantized base plus a trainable
//! low-rank adapter, `y = x · W + scale · (x · A) · B`.
//!
//! The base is never decoded to a dense weight in the forward pass — it runs
//! through the packed matmul ([`QuantizedMatrix::forward`]) — and the adapter
//! is applied to the activations, so `A · B` is never formed either. The
//! backward pass needs only `dy · Wᵀ` for the input gradient (`autodiff.rs`);
//! the base receives no gradient.
//!
//! The adapter is cortex-core's [`LoraAdapter`], so optimizers, records and
//! parameter groups treat it exactly like the adapters of cortex-core's
//! `module::Lora`. Sibling: cortex-core's `module::QLora`, which quantizes a
//! dense model in place but composes `dequantize(base) + scale · A · B` as a
//! dense weight every step; use it for models built from `Linear`, and this
//! for models loaded from packed checkpoints (`checkpoint.rs`).

use cortex_core as cortex;
use cortex_core::module::{LoraAdapter, Module, Param};
use cortex_core::tensor::{Device, Distribution, Tensor};

use crate::QuantizedMatrix;
use crate::quant::{PackedWeight, QuantError, quantize};

/// A packed linear layer with a trainable low-rank adapter.
#[derive(Module, Debug)]
pub struct QLoraLinear {
    /// The frozen packed base `W[k, n]`.
    #[module(skip)]
    base: QuantizedMatrix,
    /// The trainable factors `a[k, rank]`, `b[rank, n]` and their scale.
    pub adapter: LoraAdapter,
}

impl QLoraLinear {
    /// Attaches a rank-`rank` adapter scaled by `alpha / rank` to `base`.
    ///
    /// Standard LoRA initialization, as cortex-core's `Lora`: `a ~ N(0,
    /// 1/rank)` and `b = 0`, so the layer starts out equal to the base.
    pub fn new(base: QuantizedMatrix, rank: usize, alpha: f64, device: &Device) -> Self {
        assert!(rank > 0, "LoRA rank must be positive");
        let std = 1.0 / rank as f64;
        let a = Tensor::<2>::random([base.k(), rank], Distribution::Normal(0.0, std), device);
        let b = Tensor::<2>::zeros([rank, base.n()], device);
        Self {
            base,
            adapter: LoraAdapter {
                a: Param::from_tensor(a),
                b: Param::from_tensor(b),
                scale: alpha / rank as f64,
            },
        }
    }

    /// The frozen base.
    pub fn base(&self) -> &QuantizedMatrix {
        &self.base
    }

    /// `x[..., k]` → `[..., n]`.
    ///
    /// # Panics
    /// If the last dimension of `x` is not `k`.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dims = x.dims();
        let k = dims[D - 1];
        let rows = dims[..D - 1].iter().product::<usize>();
        let x2 = x.reshape([rows, k]);

        let base = self.base.forward(x2.clone());
        let a = self.adapter.a.val();
        let lora = x2
            .cast(a.dtype())
            .matmul(a)
            .matmul(self.adapter.b.val())
            .mul_scalar(self.adapter.scale);
        let y = base.clone() + lora.cast(base.dtype());

        let mut out = dims;
        out[D - 1] = self.base.n();
        y.reshape(out)
    }

    /// Folds the adapter into the base and re-packs it in the base's format:
    /// the deployable weight, with no adapter cost at inference.
    ///
    /// Requantization rounds the merged weight once more, so the result
    /// differs from `forward` by at most one quantization step per group.
    pub fn fuse(&self) -> Result<PackedWeight, QuantError> {
        let (k, n) = (self.base.k(), self.base.n());
        let merged = self.base.dequantize().cast(self.adapter.a.val().dtype()) + self.adapter.delta();
        let values = merged
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("merged weights read back as f32");
        quantize(&values, k, n, self.base.format())
    }
}
