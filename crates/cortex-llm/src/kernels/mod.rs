//! CubeCL implementations of [`LlmOps`](crate::LlmOps).
//!
//! Every CubeCL runtime (wgpu, Vulkan, Metal, CUDA, ROCm, CPU) shares the
//! erased `CubeBackend`, so one implementation serves them all.

mod fused;
mod qmm;

use cortex_core::backend::tensor::{FloatTensor, IntTensor};
use cortex_cubecl::CubeBackend;

use crate::LlmOps;

impl LlmOps for CubeBackend {
    fn quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        qmm::quantized_matmul(x, data, scales, biases, bits, group_size, affine)
    }

    fn dequantize_packed(
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        qmm::dequantize_packed(data, scales, biases, bits, group_size, affine)
    }

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
        qmm::gather_quantized_matmul(x, data, scales, biases, indices, bits, group_size, affine)
    }

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
        qmm::batched_quantized_matmul(x, data, scales, biases, rows, bits, group_size, affine, transpose)
    }

    fn rms_norm(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        eps: f32,
        unit_offset: bool,
    ) -> FloatTensor<Self> {
        fused::rms_norm(x, gamma, eps, unit_offset)
    }

    fn rope(
        x: FloatTensor<Self>,
        inv_freq: FloatTensor<Self>,
        offset: usize,
        traditional: bool,
    ) -> FloatTensor<Self> {
        fused::rope(x, inv_freq, offset, traditional)
    }
}
