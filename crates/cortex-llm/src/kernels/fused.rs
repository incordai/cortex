//! CubeCL kernels for RMS norm and RoPE.
//!
//! Called from `impl LlmOps for CubeBackend` (`kernels/mod.rs`); the math
//! must match `fused::rms_norm_reference` / `fused::rope_reference`, which
//! `tests/fused_ops.rs` checks on the device.

use cortex_core::backend::DType;
use cortex_core::backend::cubecl::dtype_to_storage_type;
use cortex_core::tensor::Shape;
use cortex_cubecl::{kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, prelude::*};

/// Threads cooperating on one row of the RMS norm.
const NORM_BLOCK: u32 = 256;
/// Cubes per grid row, kept under the 65535 per-dimension launch limit.
const GRID_X: usize = 32768;

/// One cube per row: a strided sum of squares into shared memory, a tree
/// reduction, then every thread scales its own elements.
#[cube(launch)]
fn rms_norm_kernel<F: Float, G: Float>(
    x: &Tensor<F>,
    gamma: &Tensor<G>,
    out: &mut Tensor<F>,
    eps: f32,
    #[comptime] block: u32,
    #[comptime] unit_offset: bool,
    #[define(F)] _x_dtype: ElemType,
    #[define(G)] _g_dtype: ElemType,
) {
    let row = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    let d = gamma.shape(0);
    let rows = x.len() / d;
    if row >= rows {
        terminate!();
    }
    let base = row * d;
    let tid = UNIT_POS_X as usize;

    let mut sum = 0.0f32;
    let mut i = tid;
    while i < d {
        let v = f32::cast_from(x[base + i]);
        sum += v * v;
        i += block as usize;
    }
    let mut partial = Shared::new_slice(block as usize);
    partial[tid] = sum;
    sync_cube();

    // A runtime counter (from `CUBE_DIM_X`, which equals `block`): a counter
    // seeded from the comptime `block` would itself be comptime, and mutating
    // it inside a runtime loop does not compile to a valid kernel.
    let mut stride = CUBE_DIM_X / 2;
    while stride > 0 {
        if (tid as u32) < stride {
            partial[tid] += partial[tid + stride as usize];
        }
        sync_cube();
        stride /= 2;
    }

    let inv = 1.0f32 / f32::sqrt(partial[0] / f32::cast_from(d) + eps);
    let mut j = tid;
    while j < d {
        let mut w = f32::cast_from(gamma[j]);
        if comptime!(unit_offset) {
            w += 1.0;
        }
        out[base + j] = F::cast_from(f32::cast_from(x[base + j]) * inv * w);
        j += block as usize;
    }
}

/// One thread per element: `out[j] = x[j] cos θ ± x[partner] sin θ`.
#[cube(launch)]
fn rope_kernel<F: Float>(
    x: &Tensor<F>,
    inv_freq: &Tensor<f32>,
    out: &mut Tensor<F>,
    offset: u32,
    seq: u32,
    #[comptime] traditional: bool,
    #[define(F)] _x_dtype: ElemType,
) {
    let idx = (CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize)
        * CUBE_DIM_X as usize
        + UNIT_POS_X as usize;
    if idx >= x.len() {
        terminate!();
    }
    let d = x.shape(x.rank() - 1);
    let half = inv_freq.shape(0);
    let rot = half * 2;
    let j = idx % d;
    if j >= rot {
        out[idx] = x[idx];
        terminate!();
    }

    let t = (idx / d) % seq as usize;
    // Interleaved pairs (2p, 2p + 1) by default; half-split pairs
    // (p, p + rot/2) replace all three below.
    let mut p = j / 2;
    let mut first = (j & 1) == 0;
    let mut partner = j ^ 1;
    if comptime!(!traditional) {
        if j < half {
            p = j;
            first = true;
            partner = j + half;
        } else {
            p = j - half;
            first = false;
            partner = j - half;
        }
    }

    let theta = f32::cast_from(offset as usize + t) * inv_freq[p];
    let mut s = f32::sin(theta);
    if first {
        s = -s;
    }
    let row = idx - j;
    let v = f32::cast_from(x[idx]) * f32::cos(theta) + f32::cast_from(x[row + partner]) * s;
    out[idx] = F::cast_from(v);
}

fn empty(like: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    let buffer = like.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(like.client.clone(), like.device.clone(), shape, buffer, dtype)
}

/// Splits `units` work items into a 2-D grid that respects the per-dimension
/// launch limit.
fn grid(units: usize) -> CubeCount {
    let x = units.clamp(1, GRID_X);
    CubeCount::Static(x as u32, units.div_ceil(x).max(1) as u32, 1)
}

pub(crate) fn rms_norm(x: CubeTensor, gamma: CubeTensor, eps: f32, unit_offset: bool) -> CubeTensor {
    let x = into_contiguous(x);
    let gamma = into_contiguous(gamma);
    let shape = x.meta.shape().clone();
    let d = shape[shape.num_dims() - 1];
    let rows = shape.num_elements() / d;
    let out = empty(&x, shape, x.dtype);
    let (x_dtype, g_dtype) = (x.dtype, gamma.dtype);
    rms_norm_kernel::launch(
        &out.client,
        grid(rows),
        CubeDim::new_1d(NORM_BLOCK),
        x.into_tensor_arg(),
        gamma.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        eps,
        NORM_BLOCK,
        unit_offset,
        dtype_to_storage_type(x_dtype),
        dtype_to_storage_type(g_dtype),
    );
    out
}

pub(crate) fn rope(x: CubeTensor, inv_freq: CubeTensor, offset: usize, traditional: bool) -> CubeTensor {
    use cortex_core::backend::ops::FloatTensorOps;
    use cortex_cubecl::CubeBackend;
    use cortex_std::FloatDType;

    let x = into_contiguous(x);
    let inv_freq = into_contiguous(CubeBackend::float_cast(inv_freq, FloatDType::F32));
    let shape = x.meta.shape().clone();
    let rank = shape.num_dims();
    let seq = shape[rank - 2];
    let elems = shape.num_elements();
    let out = empty(&x, shape, x.dtype);
    let block = 256usize;
    let x_dtype = x.dtype;
    rope_kernel::launch(
        &out.client,
        grid(elems.div_ceil(block)),
        CubeDim::new_1d(block as u32),
        x.into_tensor_arg(),
        inv_freq.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        offset as u32,
        seq as u32,
        traditional,
        dtype_to_storage_type(x_dtype),
    );
    out
}
