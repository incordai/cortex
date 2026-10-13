//! Fused normalization and rotary-embedding ops: tensor-level API, the
//! backend-generic defaults behind [`LlmOps`], and host references.
//!
//! Callers: model code, through [`rms_norm`] and [`rope`]. Both forward to
//! `LlmOps` on `Dispatch`; CubeCL backends override with the single-kernel
//! versions in `kernels/fused.rs`, every other backend composes the defaults
//! below from primitive ops. `tests/fused_ops.rs` checks every backend against
//! [`rms_norm_reference`] / [`rope_reference`].

use alloc::vec::Vec;
use cortex_core::backend::{Backend, Dispatch, TensorMetadata, tensor::FloatTensor};
use cortex_core::tensor::{DType, Device, Shape, Tensor, TensorData};
use cortex_std::FloatDType;

use crate::LlmOps;

/// `x / sqrt(mean(x², last dim) + eps) * gamma`, with `gamma` of the last
/// dimension's size. With `unit_offset` the weight applied is `1 + gamma`
/// (the convention of checkpoints that store the deviation from one).
///
/// Statistics are accumulated in f32 whatever the input dtype.
pub fn rms_norm<const D: usize>(x: Tensor<D>, gamma: Tensor<1>, eps: f32, unit_offset: bool) -> Tensor<D> {
    let d = x.dims()[D - 1];
    assert_eq!(gamma.dims()[0], d, "gamma has {} entries, the last dimension is {d}", gamma.dims()[0]);
    Tensor::from_dispatch(<Dispatch as LlmOps>::rms_norm(
        x.into_dispatch(),
        gamma.into_dispatch(),
        eps,
        unit_offset,
    ))
}

/// Rotary position embedding over the last dimension of `x[..., seq, dim]`.
///
/// Row `t` of the sequence sits at position `offset + t`. The first
/// `2 * inv_freq.len()` dimensions are rotated, pair `p` by the angle
/// `position * inv_freq[p]`; the rest pass through unchanged.
///
/// - `traditional == false`: pairs are `(p, p + rot/2)` (half-split).
/// - `traditional == true`: pairs are `(2p, 2p + 1)` (interleaved).
///
/// Build `inv_freq` with [`rope_inv_freq`] for the standard schedule, or
/// pass any schedule (linear / NTK / YaRN-scaled frequencies) directly.
pub fn rope<const D: usize>(x: Tensor<D>, inv_freq: Tensor<1>, offset: usize, traditional: bool) -> Tensor<D> {
    assert!(D >= 2, "rope needs [..., seq, dim]");
    let d = x.dims()[D - 1];
    let rot = 2 * inv_freq.dims()[0];
    assert!(rot <= d, "rotating {rot} dimensions of a {d}-dimensional head");
    Tensor::from_dispatch(<Dispatch as LlmOps>::rope(
        x.into_dispatch(),
        inv_freq.into_dispatch(),
        offset,
        traditional,
    ))
}

/// The standard RoPE schedule `inv_freq[p] = base^(-2p / rotary_dims)`.
pub fn rope_inv_freq(rotary_dims: usize, base: f32, device: &Device) -> Tensor<1> {
    assert!(rotary_dims.is_multiple_of(2), "rotary dimensions must be even");
    let freqs: Vec<f32> = (0..rotary_dims / 2)
        .map(|p| base.powf(-2.0 * p as f32 / rotary_dims as f32))
        .collect();
    Tensor::from_data(TensorData::new(freqs, [rotary_dims / 2]), (device, DType::F32))
}

/// The partner of every dimension: the other half of its rotated pair, or
/// itself for a pass-through dimension. Shared by the default and the
/// reference so both pair dimensions identically.
pub(crate) fn rope_partner(d: usize, rot: usize, traditional: bool) -> Vec<usize> {
    let half = rot / 2;
    (0..d)
        .map(|j| {
            if j >= rot {
                j
            } else if traditional {
                j ^ 1
            } else if j < half {
                j + half
            } else {
                j - half
            }
        })
        .collect()
}

/// Pair index of a rotated dimension, and whether it is the first of its pair.
pub(crate) fn rope_pair(j: usize, rot: usize, traditional: bool) -> (usize, bool) {
    if traditional {
        (j / 2, j.is_multiple_of(2))
    } else {
        let half = rot / 2;
        if j < half { (j, true) } else { (j - half, false) }
    }
}

/// Generic RMS norm from primitive ops; see [`rms_norm`].
pub(crate) fn rms_norm_default<B: Backend>(
    x: FloatTensor<B>,
    gamma: FloatTensor<B>,
    eps: f32,
    unit_offset: bool,
) -> FloatTensor<B> {
    let dtype = FloatDType::from(x.dtype());
    let shape = x.shape();
    let rank = shape.num_dims();
    let x = B::float_cast(x, FloatDType::F32);
    let ms = B::float_mean_dim(B::float_mul(x.clone(), x.clone()), rank - 1);
    let inv = B::float_recip(B::float_sqrt(B::float_add_scalar(ms, eps.into())));
    let y = B::float_mul(x, inv);

    let mut g_shape = alloc::vec![1usize; rank];
    g_shape[rank - 1] = shape[rank - 1];
    let mut g = B::float_cast(gamma, FloatDType::F32);
    if unit_offset {
        g = B::float_add_scalar(g, 1.0f32.into());
    }
    let g = B::float_reshape(g, Shape::from(g_shape));
    B::float_cast(B::float_mul(y, g), dtype)
}

/// Generic RoPE from primitive ops; see [`rope`].
///
/// `out = x * cos + x[partner] * signed_sin`, where `signed_sin` is `-sin`
/// on the first dimension of a pair, `+sin` on the second, and `cos = 1`,
/// `sin = 0` on pass-through dimensions.
pub(crate) fn rope_default<B: Backend>(
    x: FloatTensor<B>,
    inv_freq: FloatTensor<B>,
    offset: usize,
    traditional: bool,
) -> FloatTensor<B> {
    let device = x.device();
    let dtype = FloatDType::from(x.dtype());
    let shape = x.shape();
    let rank = shape.num_dims();
    let (s, d) = (shape[rank - 2], shape[rank - 1]);
    let half = inv_freq.shape()[0];
    let rot = 2 * half;

    let positions: Vec<f32> = (0..s).map(|t| (offset + t) as f32).collect();
    let positions = B::float_from_data(TensorData::new(positions, [s, 1]), &device);
    let inv = B::float_reshape(B::float_cast(inv_freq, FloatDType::F32), Shape::new([1, half]));
    let theta = B::float_mul(positions, inv); // [s, half]
    let cos = B::float_cos(theta.clone());
    let sin = B::float_sin(theta);
    let neg_sin = B::float_neg(sin.clone());

    // Expand the per-pair tables to per-dimension tables in the pair layout.
    let (cos_rot, sin_rot) = if traditional {
        let pairwise = |a: FloatTensor<B>, b: FloatTensor<B>| {
            let a = B::float_reshape(a, Shape::new([s, half, 1]));
            let b = B::float_reshape(b, Shape::new([s, half, 1]));
            B::float_reshape(B::float_cat(alloc::vec![a, b], 2), Shape::new([s, rot]))
        };
        (pairwise(cos.clone(), cos), pairwise(neg_sin, sin))
    } else {
        (
            B::float_cat(alloc::vec![cos.clone(), cos], 1),
            B::float_cat(alloc::vec![neg_sin, sin], 1),
        )
    };
    let (cos_full, sin_full) = if rot < d {
        let rest = Shape::new([s, d - rot]);
        (
            B::float_cat(
                alloc::vec![cos_rot, B::float_ones(rest.clone(), &device, FloatDType::F32)],
                1,
            ),
            B::float_cat(
                alloc::vec![sin_rot, B::float_zeros(rest, &device, FloatDType::F32)],
                1,
            ),
        )
    } else {
        (cos_rot, sin_rot)
    };
    let mut table_shape = alloc::vec![1usize; rank];
    table_shape[rank - 2] = s;
    table_shape[rank - 1] = d;
    let cos_full = B::float_reshape(cos_full, Shape::from(table_shape.clone()));
    let sin_full = B::float_reshape(sin_full, Shape::from(table_shape));

    let partner: Vec<i64> = rope_partner(d, rot, traditional)
        .into_iter()
        .map(|p| p as i64)
        .collect();
    let partner = B::int_from_data(TensorData::new(partner, [d]), &device);

    let x = B::float_cast(x, FloatDType::F32);
    let swapped = B::float_select(x.clone(), rank - 1, partner);
    let out = B::float_add(B::float_mul(x, cos_full), B::float_mul(swapped, sin_full));
    B::float_cast(out, dtype)
}

/// Host reference for [`rms_norm`] on a row-major `[rows, d]` buffer.
pub fn rms_norm_reference(x: &[f32], gamma: &[f32], eps: f32, unit_offset: bool) -> Vec<f32> {
    let d = gamma.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(d) {
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for (v, g) in row.iter().zip(gamma) {
            let w = if unit_offset { 1.0 + *g as f64 } else { *g as f64 };
            out.push((*v as f64 * inv * w) as f32);
        }
    }
    out
}

/// Host reference for [`rope`] on a row-major `[..., s, d]` buffer.
pub fn rope_reference(
    x: &[f32],
    s: usize,
    d: usize,
    inv_freq: &[f32],
    offset: usize,
    traditional: bool,
) -> Vec<f32> {
    let rot = 2 * inv_freq.len();
    let partner = rope_partner(d, rot, traditional);
    let mut out = Vec::with_capacity(x.len());
    for (r, row) in x.chunks(d).enumerate() {
        let pos = (offset + r % s) as f64;
        for j in 0..d {
            if j >= rot {
                out.push(row[j]);
                continue;
            }
            let (p, first) = rope_pair(j, rot, traditional);
            let theta = pos * inv_freq[p] as f64;
            let sign = if first { -1.0 } else { 1.0 };
            let v = row[j] as f64 * theta.cos() + row[partner[j]] as f64 * sign * theta.sin();
            out.push(v as f32);
        }
    }
    out
}
