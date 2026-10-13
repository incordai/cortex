//! Backend operations supplied by this crate's extension.
//!
//! Every method has a backend-generic default built from primitive tensor
//! ops, so any backend runs them correctly. CubeCL backends override them with
//! the kernels in `kernels/`, and the fusion decorator (`fusion.rs`) registers
//! each call as one custom operation so it can sit inside a fused stream.
//!
//! Callers: the tensor-level API in `quantized.rs` and `fused.rs`, which
//! validates shapes and formats before anything reaches this trait.

use alloc::vec::Vec;
use cortex_core as cortex;
use cortex_core::backend::{
    Backend, TensorMetadata, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use cortex_core::tensor::{Shape, TensorData};
use cortex_std::{FloatDType, IntDType, Slice};

/// LLM inference operations supplied by a backend extension.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(any(
        feature = "wgpu",
        feature = "webgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "cpu"
    )),
    NdArray: cfg(feature = "ndarray"),
    LibTorch: cfg(feature = "tch"),
    Remote: cfg(feature = "remote"),
    Capture: cfg(feature = "capture"),
    Autodiff: cfg(feature = "autodiff"),
)]
pub trait LlmOps: Backend {
    /// `x[m, k] · W[k, n]` for a packed weight (see [`crate::quant`]).
    ///
    /// `data` is `[k, n * bits / 32]` of `u32` words, `scales` and `biases`
    /// are `[k, n / group_size]`. In symmetric mode (`affine == false`)
    /// `biases` is not read; pass `scales` again.
    fn quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        let dtype = FloatDType::from(x.dtype());
        let w = decode_packed::<Self>(data, scales, biases, bits, group_size, affine, dtype);
        Self::float_matmul(x, w)
    }

    /// Decodes a packed `[rows, n * bits / 32]` weight to dense `[rows, n]`
    /// values in the dtype of `scales`. Arguments as in
    /// [`quantized_matmul`](LlmOps::quantized_matmul).
    ///
    /// Callers: the quantized matmul backward passes in `autodiff.rs`, which
    /// need `Wᵀ` to propagate the input gradient.
    fn dequantize_packed(
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        let dtype = FloatDType::from(scales.dtype());
        decode_packed::<Self>(data, scales, biases, bits, group_size, affine, dtype)
    }

    /// Mixture-of-experts matmul: row `t` of `x[t, k]` is multiplied by the
    /// expert `indices[t]` of a stacked packed weight `W[e, k, n]`.
    ///
    /// `data` is `[e, k, n * bits / 32]`, `scales` and `biases` are
    /// `[e, k, n / group_size]`, `indices` is `[t]`. Only the selected experts
    /// are decoded.
    #[allow(clippy::too_many_arguments)]
    fn gather_quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        indices: IntTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        let dtype = FloatDType::from(x.dtype());
        let x_shape = x.shape();
        let (t, k) = (x_shape[0], x_shape[1]);
        let d_shape = data.shape();
        let wpr = d_shape[2];
        let gpr = scales.shape()[2];
        let n = wpr * 32 / bits as usize;

        // Per-token copies of the selected experts, flattened to one tall
        // packed matrix so the 2-D decoder can be reused.
        let data = Self::int_select(data, 0, indices.clone());
        let scales = Self::float_select(scales, 0, indices.clone());
        let biases = Self::float_select(biases, 0, indices);
        let data = Self::int_reshape(data, Shape::new([t * k, wpr]));
        let scales = Self::float_reshape(scales, Shape::new([t * k, gpr]));
        let biases = Self::float_reshape(biases, Shape::new([t * k, gpr]));

        let w = decode_packed::<Self>(data, scales, biases, bits, group_size, affine, dtype);
        let w = Self::float_reshape(w, Shape::new([t, k, n]));
        let x = Self::float_reshape(x, Shape::new([t, 1, k]));
        let y = Self::float_matmul(x, w);
        Self::float_reshape(y, Shape::new([t, n]))
    }

    /// One packed matrix per batch entry: `x[b, m, ·]` times the first `rows`
    /// rows of `data[b]` (`[b, capacity, words]`, `capacity >= rows`;
    /// `scales` and `biases` are `[b, capacity, groups]`), so a growing
    /// buffer is read in place.
    ///
    /// Without `transpose`, `W[b]` is `[rows, n]` and the result `x · W[b]`
    /// (`x` has `rows` columns), as in [`quantized_matmul`](LlmOps::quantized_matmul).
    /// With `transpose`, `W[b]` is `[rows, k]`, packed along the `k` that
    /// `x` contracts, and the result is `x · W[b]ᵀ`, `[b, m, rows]`.
    ///
    /// Callers: `QuantizedKvCache::attend` in `qcache.rs` — the scores
    /// against packed keys (transposed) and the weighted sum of packed values.
    #[allow(clippy::too_many_arguments)]
    fn batched_quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        rows: usize,
        bits: u32,
        group_size: usize,
        affine: bool,
        transpose: bool,
    ) -> FloatTensor<Self> {
        let dtype = FloatDType::from(x.dtype());
        let w = decode_batched::<Self>(data, scales, biases, rows, bits, group_size, affine, dtype);
        let w = if transpose { Self::float_transpose(w) } else { w };
        Self::float_matmul(x, w)
    }

    /// RMS normalization over the last dimension; see [`crate::rms_norm`].
    fn rms_norm(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        eps: f32,
        unit_offset: bool,
    ) -> FloatTensor<Self> {
        crate::fused::rms_norm_default::<Self>(x, gamma, eps, unit_offset)
    }

    /// Rotary position embedding of `x[..., seq, dim]`; see [`crate::rope`].
    fn rope(
        x: FloatTensor<Self>,
        inv_freq: FloatTensor<Self>,
        offset: usize,
        traditional: bool,
    ) -> FloatTensor<Self> {
        crate::fused::rope_default::<Self>(x, inv_freq, offset, traditional)
    }
}

/// Decodes a packed `[rows, words]` matrix to `[rows, n]` values on the
/// device, with primitive ops only.
///
/// Mirrors `quant::code_at` and `quant::dequantize`. Codes are widened to
/// `i64` first so shifts never meet a sign bit: a code that straddles two
/// words is `(lo >> off) | (hi << (32 - off))`, masked to `bits`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_packed<B: Backend>(
    data: IntTensor<B>,
    scales: FloatTensor<B>,
    biases: FloatTensor<B>,
    bits: u32,
    group_size: usize,
    affine: bool,
    dtype: FloatDType,
) -> FloatTensor<B> {
    let device = data.device();
    let d_shape = data.shape();
    let (rows, wpr) = (d_shape[0], d_shape[1]);
    let n = wpr * 32 / bits as usize;
    let gpr = n / group_size;

    let mut lo_idx = Vec::with_capacity(n);
    let mut lo_shift = Vec::with_capacity(n);
    let mut hi_idx = Vec::with_capacity(n);
    let mut hi_shift = Vec::with_capacity(n);
    let mut hi_keep = Vec::with_capacity(n);
    for j in 0..n {
        let pos = j * bits as usize;
        let word = pos / 32;
        let off = (pos % 32) as u32;
        let straddle = off + bits > 32;
        lo_idx.push(word as i64);
        lo_shift.push(off as i64);
        hi_idx.push(if straddle { word as i64 + 1 } else { word as i64 });
        hi_shift.push(if straddle { (32 - off) as i64 } else { 0 });
        hi_keep.push(straddle as i64);
    }
    let per_column = |values: Vec<i64>| {
        let t = B::int_from_data(TensorData::new(values, [1, n]), &device);
        let t = B::int_cast(t, IntDType::I64);
        B::int_expand(t, Shape::new([rows, n]))
    };

    let words = B::int_cast(data, IntDType::I64);
    let lo = B::int_gather(1, words.clone(), per_column(lo_idx));
    let lo = B::bitwise_right_shift(lo, per_column(lo_shift));
    let hi = B::int_gather(1, words, per_column(hi_idx));
    let hi = B::bitwise_left_shift(hi, per_column(hi_shift));
    let hi = B::int_mul(hi, per_column(hi_keep));
    let codes = B::bitwise_and_scalar(B::bitwise_or(lo, hi), ((1i64 << bits) - 1).into());
    let q = B::int_into_float(codes, dtype);

    let per_group = |t: FloatTensor<B>| {
        let t = B::float_reshape(t, Shape::new([rows, gpr, 1]));
        let t = B::float_expand(t, Shape::new([rows, gpr, group_size]));
        B::float_reshape(t, Shape::new([rows, n]))
    };
    let s = per_group(B::float_cast(scales, dtype));
    let b = if affine {
        per_group(B::float_cast(biases, dtype))
    } else {
        B::float_mul_scalar(s.clone(), (-((1u32 << (bits - 1)) as f32)).into())
    };
    B::float_add(B::float_mul(q, s), b)
}

/// The first `rows` rows of each packed `data[b]` (`[b, capacity, words]`)
/// decoded to `[b, rows, n]` in `dtype`.
///
/// Callers: the `batched_quantized_matmul` default above and its backward
/// pass in `autodiff.rs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_batched<B: Backend>(
    data: IntTensor<B>,
    scales: FloatTensor<B>,
    biases: FloatTensor<B>,
    rows: usize,
    bits: u32,
    group_size: usize,
    affine: bool,
    dtype: FloatDType,
) -> FloatTensor<B> {
    let d_shape = data.shape();
    let (batch, wpr) = (d_shape[0], d_shape[2]);
    let gpr = scales.shape()[2];
    let n = wpr * 32 / bits as usize;
    let used = [Slice::from(0..batch), Slice::from(0..rows), Slice::from(..)];
    let data = B::int_reshape(B::int_slice(data, &used), Shape::new([batch * rows, wpr]));
    let scales = B::float_reshape(B::float_slice(scales, &used), Shape::new([batch * rows, gpr]));
    let biases = B::float_reshape(B::float_slice(biases, &used), Shape::new([batch * rows, gpr]));
    let w = decode_packed::<B>(data, scales, biases, bits, group_size, affine, dtype);
    B::float_reshape(w, Shape::new([batch, rows, n]))
}

#[allow(unused_macros)]
macro_rules! impl_llm_ops {
    ($backend:ty) => {
        impl LlmOps for $backend {}
    };
}

#[cfg(feature = "flex")]
impl_llm_ops!(cortex_core::backend::Flex);
#[cfg(feature = "ndarray")]
impl_llm_ops!(cortex_core::backend::NdArray);
#[cfg(feature = "tch")]
impl_llm_ops!(cortex_core::backend::LibTorch);
#[cfg(feature = "router")]
impl<C: cortex_router::RouterChannel> LlmOps for cortex_router::BackendRouter<C> {}
