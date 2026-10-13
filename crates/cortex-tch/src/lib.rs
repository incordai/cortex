#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::single_range_in_vec_init)]
#![deprecated(
    since = "0.22.0",
    note = "cortex-tch is deprecated and will be removed in a future release. For GPU acceleration use a CubeCL backend: cortex-cuda (NVIDIA), cortex-rocm (AMD), or cortex-wgpu (Metal, Vulkan, WebGPU). For CPU execution use cortex-cpu (CubeCL) or cortex-flex (pure Rust)."
)]

//! Cortex Tch Backend
//!
//! **Deprecated:** This backend is deprecated and will be removed in a future release.
//! Please migrate to one of the actively maintained backends:
//! - CubeCL GPU backends: `cortex-cuda` (NVIDIA), `cortex-rocm` (AMD), `cortex-wgpu` (Metal, Vulkan,
//!   WebGPU)
//! - CPU backends: `cortex-cpu` (CubeCL) or `cortex-flex` (pure Rust, std/no_std/WASM)

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;
