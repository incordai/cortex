//! CubeCL kernels for the packed quantized matmuls.
//!
//! Called from `impl LlmOps for CubeBackend` (`kernels/mod.rs`). Decoding
//! must agree with `quant::code_at` bit for bit; `tests/quantized_matmul.rs`
//! checks both against the host reference.
//!
//! # `qmm_kernel` — decode and small batches (`m <= MAX_ROWS`)
//!
//! One thread owns eight adjacent output columns over a slice of the `k`
//! rows, so neighbouring threads read neighbouring codes of a row (coalesced)
//! and the packed weight is read exactly once per launch whatever `m` is: the
//! eight decoded codes of a row are reused for every input row. Eight columns
//! always share one scale because group sizes are multiples of eight. Slices
//! write partial sums `[slices, rows, n]`, reduced afterwards.
//!
//! The same kernel serves the gathered (MoE) form with one input row per
//! `z` index and the expert offset read from `indices`, and the batched form
//! (`batched_quantized_matmul`) with `rows_m` input rows per `z`, matrix `z`.
//!
//! # `qmm_t_kernel` — against `Wᵀ` (attention scores over packed keys)
//!
//! When `x` contracts the packed dimension, one thread owns one row of `W`
//! (one cached position) and decodes it once for all `rows_m` inputs, using
//! `Σ x·(s·q + b) = s·Σ x·q + b·Σ x` per group.
//!
//! # `dequant_kernel` — prefill
//!
//! Larger batches are compute-bound, so the weight is decoded once into a
//! dense matrix and handed to the backend's tiled matmul.

use cortex_core::backend::cubecl::dtype_to_storage_type;
use cortex_core::backend::{DType, ops::FloatTensorOps};
use cortex_core::tensor::Shape;
use cortex_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cortex_std::FloatDType;
use cubecl::{CubeCount, CubeDim, prelude::*};

/// Largest `m` handled by `qmm_kernel`; beyond it the weight is decoded once
/// and the dense matmul is cheaper per row.
pub(crate) const MAX_ROWS: usize = 8;

/// Code of column `j` in the packed row starting at word `row`.
#[cube]
fn code_at(data: &Tensor<u32>, row: usize, j: usize, #[comptime] bits: u32) -> u32 {
    let mask = (1u32 << bits) - 1;
    let pos = j * bits as usize;
    let word = row + pos / 32;
    let off = (pos % 32) as u32;
    let mut v = data[word] >> off;
    if comptime!(32 % bits != 0) {
        // 3-, 5- and 6-bit codes may straddle two words.
        if off + bits > 32 {
            v |= data[word + 1] << (32 - off);
        }
    }
    v & mask
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn qmm_kernel<F: Float, S: Float>(
    x: &Tensor<F>,
    data: &Tensor<u32>,
    scales: &Tensor<S>,
    biases: &Tensor<S>,
    indices: &Tensor<u32>,
    partial: &mut Tensor<f32>,
    #[comptime] bits: u32,
    #[comptime] group_size: u32,
    #[comptime] affine: bool,
    #[comptime] rows_m: u32,
    #[comptime] gather: bool,
    #[define(F)] _x_dtype: ElemType,
    #[define(S)] _s_dtype: ElemType,
) {
    let c8 = ABSOLUTE_POS_X as usize;
    let slice = ABSOLUTE_POS_Y as usize;
    let z = ABSOLUTE_POS_Z as usize;
    let slices = partial.shape(0);
    let out_rows = partial.shape(1);
    let n = partial.shape(2);
    if c8 * 8 >= n || slice >= slices {
        terminate!();
    }
    let k = x.shape(1);
    let wpr = data.shape(data.rank() - 1);
    let gpr = scales.shape(scales.rank() - 1);
    // Rows per stacked matrix: `k` for experts, a cache's capacity when
    // only its first `k` rows are in use.
    let w_rows = data.shape(data.rank() - 2);

    // A batched launch multiplies its `z`-th rows by matrix `z`; a plain one
    // has the single `z == 0`.
    let mut expert = z;
    if comptime!(gather) {
        expert = indices[z] as usize;
    }
    let w_base = expert * w_rows * wpr;
    let s_base = expert * w_rows * gpr;
    let group = c8 * 8 / group_size as usize;
    let offset = f32::cast_from(1u32 << (bits - 1));

    let rows = k.div_ceil(slices);
    let r0 = slice * rows;
    let mut r1 = r0 + rows;
    if r1 > k {
        r1 = k;
    }

    let rows8 = rows_m * 8;
    let mut acc = Array::<f32>::new(rows8 as usize);
    let mut acc_bias = Array::<f32>::new(rows_m as usize);
    #[unroll]
    for i in 0..rows8 {
        acc[i as usize] = 0.0;
    }
    #[unroll]
    for i in 0..rows_m {
        acc_bias[i as usize] = 0.0;
    }

    let mut q = Array::<f32>::new(8usize);
    for r in r0..r1 {
        let s = f32::cast_from(scales[s_base + r * gpr + group]);
        let mut b = -s * offset;
        if comptime!(affine) {
            b = f32::cast_from(biases[s_base + r * gpr + group]);
        }
        let row = w_base + r * wpr;
        #[unroll]
        for i in 0..8u32 {
            q[i as usize] = f32::cast_from(code_at(data, row, c8 * 8 + i as usize, bits));
        }
        #[unroll]
        for mm in 0..rows_m {
            // Gathered launches have `rows_m == 1` and one `z` per input row,
            // batched ones `rows_m` rows per `z`, plain ones a single `z == 0`.
            let xr = z * rows_m as usize + mm as usize;
            let xv = f32::cast_from(x[xr * k + r]);
            let xs = xv * s;
            acc_bias[mm as usize] += xv * b;
            #[unroll]
            for i in 0..8u32 {
                acc[(mm * 8 + i) as usize] += xs * q[i as usize];
            }
        }
    }

    #[unroll]
    for mm in 0..rows_m {
        let out_row = z * rows_m as usize + mm as usize;
        let base = (slice * out_rows + out_row) * n + c8 * 8;
        #[unroll]
        for i in 0..8u32 {
            partial[base + i as usize] = acc[(mm * 8 + i) as usize] + acc_bias[mm as usize];
        }
    }
}

#[cube(launch)]
fn dequant_kernel<F: Float, S: Float>(
    data: &Tensor<u32>,
    scales: &Tensor<S>,
    biases: &Tensor<S>,
    out: &mut Tensor<F>,
    #[comptime] bits: u32,
    #[comptime] group_size: u32,
    #[comptime] affine: bool,
    #[define(F)] _x_dtype: ElemType,
    #[define(S)] _s_dtype: ElemType,
) {
    let c8 = ABSOLUTE_POS_X as usize;
    let r = ABSOLUTE_POS_Y as usize;
    let k = out.shape(0);
    let n = out.shape(1);
    if c8 * 8 >= n || r >= k {
        terminate!();
    }
    let wpr = data.shape(1);
    let gpr = scales.shape(1);
    let group = c8 * 8 / group_size as usize;
    let s = f32::cast_from(scales[r * gpr + group]);
    let mut b = -s * f32::cast_from(1u32 << (bits - 1));
    if comptime!(affine) {
        b = f32::cast_from(biases[r * gpr + group]);
    }
    #[unroll]
    for i in 0..8u32 {
        let col = c8 * 8 + i as usize;
        let code = f32::cast_from(code_at(data, r * wpr, col, bits));
        out[r * n + col] = F::cast_from(s * code + b);
    }
}

fn empty(like: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    let buffer = like.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(like.client.clone(), like.device.clone(), shape, buffer, dtype)
}

/// Row slices for `k` rows: enough to occupy the device, each >= 64 rows.
fn slices_for(k: usize) -> usize {
    (k / 64).clamp(1, 64)
}

#[allow(clippy::too_many_arguments)]
fn launch_qmm(
    x: CubeTensor,
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    indices: Option<CubeTensor>,
    out_rows: usize,
    rows_m: usize,
    n: usize,
    bits: u32,
    group_size: usize,
    affine: bool,
) -> CubeTensor {
    let x_dtype = x.dtype;
    let k = x.meta.shape()[1];
    let slices = slices_for(k);
    let partial = empty(&x, Shape::new([slices, out_rows, n]), DType::F32);
    let gather = indices.is_some();
    // One `z` per group of `rows_m` input rows: one per row when gathering,
    // one per matrix when batched, one in all for a plain launch.
    let z = out_rows / rows_m;
    let rows_m = rows_m as u32;
    let indices = indices.unwrap_or_else(|| data.clone());

    let cube_dim = CubeDim::new_3d(64, 4, 1);
    let count = CubeCount::Static(
        (n / 8).div_ceil(64) as u32,
        slices.div_ceil(4) as u32,
        z as u32,
    );
    let s_dtype = dtype_to_storage_type(scales.dtype);
    qmm_kernel::launch(
        &partial.client,
        count,
        cube_dim,
        x.into_tensor_arg(),
        data.into_tensor_arg(),
        scales.into_tensor_arg(),
        biases.into_tensor_arg(),
        indices.into_tensor_arg(),
        partial.clone().into_tensor_arg(),
        bits,
        group_size as u32,
        affine,
        rows_m,
        gather,
        dtype_to_storage_type(x_dtype),
        s_dtype,
    );

    let summed = CubeBackend::float_sum_dim(partial, 0);
    let summed = CubeBackend::float_reshape(summed, Shape::new([out_rows, n]));
    if x_dtype == DType::F32 {
        summed
    } else {
        CubeBackend::float_cast(summed, FloatDType::from(x_dtype))
    }
}

/// Decodes a packed `[k, words]` weight into a dense `[k, n]` matrix.
fn dequantize(
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    bits: u32,
    group_size: usize,
    affine: bool,
    dtype: DType,
) -> CubeTensor {
    let shape = data.meta.shape().clone();
    let (k, n) = (shape[0], shape[1] * 32 / bits as usize);
    let out = empty(&data, Shape::new([k, n]), dtype);
    let cube_dim = CubeDim::new_3d(64, 4, 1);
    let count = CubeCount::Static((n / 8).div_ceil(64) as u32, k.div_ceil(4) as u32, 1);
    let s_dtype = dtype_to_storage_type(scales.dtype);
    dequant_kernel::launch(
        &out.client,
        count,
        cube_dim,
        data.into_tensor_arg(),
        scales.into_tensor_arg(),
        biases.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        bits,
        group_size as u32,
        affine,
        dtype_to_storage_type(dtype),
        s_dtype,
    );
    out
}

/// A packed weight decoded to `[k, n]` in the dtype of `scales`; the CubeCL
/// `LlmOps::dequantize` (used by the backward passes in `autodiff.rs`).
pub(crate) fn dequantize_packed(
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    bits: u32,
    group_size: usize,
    affine: bool,
) -> CubeTensor {
    let dtype = scales.dtype;
    dequantize(
        into_contiguous(data),
        into_contiguous(scales),
        into_contiguous(biases),
        bits,
        group_size,
        affine,
        dtype,
    )
}

/// `x[m, k] · W[k, n]` on a CubeCL device.
pub(crate) fn quantized_matmul(
    x: CubeTensor,
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    bits: u32,
    group_size: usize,
    affine: bool,
) -> CubeTensor {
    let x = into_contiguous(x);
    let data = into_contiguous(data);
    let scales = into_contiguous(scales);
    let biases = into_contiguous(biases);
    let m = x.meta.shape()[0];
    let n = data.meta.shape()[1] * 32 / bits as usize;
    if m <= MAX_ROWS {
        launch_qmm(x, data, scales, biases, None, m, m, n, bits, group_size, affine)
    } else {
        let w = dequantize(data, scales, biases, bits, group_size, affine, x.dtype);
        CubeBackend::float_matmul(x, w)
    }
}

/// Row `t` of `x[t, k]` times expert `indices[t]` of `W[e, k, n]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_quantized_matmul(
    x: CubeTensor,
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    indices: CubeTensor,
    bits: u32,
    group_size: usize,
    affine: bool,
) -> CubeTensor {
    use cortex_core::backend::ops::IntTensorOps;
    use cortex_std::IntDType;

    let x = into_contiguous(x);
    let data = into_contiguous(data);
    let scales = into_contiguous(scales);
    let biases = into_contiguous(biases);
    let indices = into_contiguous(CubeBackend::int_cast(indices, IntDType::U32));
    let t = x.meta.shape()[0];
    let n = data.meta.shape()[2] * 32 / bits as usize;
    launch_qmm(x, data, scales, biases, Some(indices), t, 1, n, bits, group_size, affine)
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn qmm_t_kernel<F: Float, S: Float>(
    x: &Tensor<F>,
    data: &Tensor<u32>,
    scales: &Tensor<S>,
    biases: &Tensor<S>,
    out: &mut Tensor<f32>,
    #[comptime] bits: u32,
    #[comptime] group_size: u32,
    #[comptime] affine: bool,
    #[comptime] rows_m: u32,
    #[define(F)] _x_dtype: ElemType,
    #[define(S)] _s_dtype: ElemType,
) {
    let col = ABSOLUTE_POS_X as usize;
    let z = ABSOLUTE_POS_Y as usize;
    let n = out.shape(1);
    if col >= n || z >= data.shape(0) {
        terminate!();
    }
    let k = x.shape(1);
    let capacity = data.shape(1);
    let wpr = data.shape(2);
    let gpr = scales.shape(2);
    let row = (z * capacity + col) * wpr;
    let srow = (z * capacity + col) * gpr;
    let offset = f32::cast_from(1u32 << (bits - 1));
    let x0 = z * rows_m as usize;

    let mut acc = Array::<f32>::new(rows_m as usize);
    let mut dot = Array::<f32>::new(rows_m as usize);
    let mut xsum = Array::<f32>::new(rows_m as usize);
    #[unroll]
    for mm in 0..rows_m {
        acc[mm as usize] = 0.0;
    }
    for g in 0..gpr {
        let s = f32::cast_from(scales[srow + g]);
        let mut b = -s * offset;
        if comptime!(affine) {
            b = f32::cast_from(biases[srow + g]);
        }
        #[unroll]
        for mm in 0..rows_m {
            dot[mm as usize] = 0.0;
            xsum[mm as usize] = 0.0;
        }
        // Within a group, Σ x·(s·q + b) = s·Σ x·q + b·Σ x.
        let k0 = g * group_size as usize;
        for j in 0..group_size as usize {
            let kk = k0 + j;
            let q = f32::cast_from(code_at(data, row, kk, bits));
            #[unroll]
            for mm in 0..rows_m {
                let xv = f32::cast_from(x[(x0 + mm as usize) * k + kk]);
                dot[mm as usize] += xv * q;
                xsum[mm as usize] += xv;
            }
        }
        #[unroll]
        for mm in 0..rows_m {
            acc[mm as usize] += s * dot[mm as usize] + b * xsum[mm as usize];
        }
    }
    #[unroll]
    for mm in 0..rows_m {
        out[(x0 + mm as usize) * n + col] = acc[mm as usize];
    }
}

/// `x[b·m, k] · W[b]ᵀ` for `W[b]` the first `rows` rows of `data[b]`
/// (`[b, capacity, k·bits/32]`): one thread per (row of `W`, matrix), each
/// decoding its row once for all `m` inputs.
#[allow(clippy::too_many_arguments)]
fn launch_qmm_t(
    x: CubeTensor,
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    m: usize,
    rows: usize,
    bits: u32,
    group_size: usize,
    affine: bool,
) -> CubeTensor {
    let x_dtype = x.dtype;
    let batch = data.meta.shape()[0];
    let out = empty(&x, Shape::new([batch * m, rows]), DType::F32);
    let cube_dim = CubeDim::new_3d(64, 4, 1);
    let count = CubeCount::Static(rows.div_ceil(64) as u32, batch.div_ceil(4) as u32, 1);
    let s_dtype = dtype_to_storage_type(scales.dtype);
    qmm_t_kernel::launch(
        &out.client,
        count,
        cube_dim,
        x.into_tensor_arg(),
        data.into_tensor_arg(),
        scales.into_tensor_arg(),
        biases.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        bits,
        group_size as u32,
        affine,
        m as u32,
        dtype_to_storage_type(x_dtype),
        s_dtype,
    );
    if x_dtype == DType::F32 {
        out
    } else {
        CubeBackend::float_cast(out, FloatDType::from(x_dtype))
    }
}

/// `x[b, m, ·]` times the first `rows` rows of each packed `data[b]`, as
/// `LlmOps::batched_quantized_matmul` describes. Up to [`MAX_ROWS`] inputs
/// per matrix read the packed rows in place — the cache is never decoded —
/// with `qmm_t_kernel` (transposed) or the batched `qmm_kernel`; more decode
/// the used rows once for the dense batched matmul.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_quantized_matmul(
    x: CubeTensor,
    data: CubeTensor,
    scales: CubeTensor,
    biases: CubeTensor,
    rows: usize,
    bits: u32,
    group_size: usize,
    affine: bool,
    transpose: bool,
) -> CubeTensor {
    let x = into_contiguous(x);
    let data = into_contiguous(data);
    let scales = into_contiguous(scales);
    let biases = into_contiguous(biases);
    let x_shape = x.meta.shape().clone();
    let (batch, m, k) = (x_shape[0], x_shape[1], x_shape[2]);
    let cols = data.meta.shape()[2] * 32 / bits as usize;

    if m > MAX_ROWS {
        let dtype = FloatDType::from(x.dtype);
        let w = crate::ops::decode_batched::<CubeBackend>(data, scales, biases, rows, bits, group_size, affine, dtype);
        let w = if transpose { CubeBackend::float_transpose(w) } else { w };
        return CubeBackend::float_matmul(x, w);
    }
    let flat = CubeBackend::float_reshape(x, Shape::new([batch * m, k]));
    let (y, n) = if transpose {
        (launch_qmm_t(flat, data, scales, biases, m, rows, bits, group_size, affine), rows)
    } else {
        (launch_qmm(flat, data, scales, biases, None, batch * m, m, cols, bits, group_size, affine), cols)
    };
    CubeBackend::float_reshape(y, Shape::new([batch, m, n]))
}
