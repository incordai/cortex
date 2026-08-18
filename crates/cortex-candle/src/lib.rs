#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unused)] // TODO remove when backend filled
#![deprecated(
    since = "0.21.0",
    note = "cortex-candle is deprecated and will be removed in a future release. Use cortex-cubecl (CUDA/ROCm/Vulkan/Metal/WebGPU), cortex-flex, or cortex-tch instead."
)]

//! Cortex Candle Backend
//!
//! **Deprecated:** This backend is deprecated and will be removed in a future release.
//! Please migrate to one of the actively maintained backends:
//! - CubeCL backends (CUDA, ROCm, Vulkan, Metal, WebGPU) for GPU acceleration
//! - Flex (`cortex-flex`) for portable pure-Rust CPU execution (std, no_std, WASM)
//! - LibTorch (`cortex-tch`) for a mature CPU/GPU backend

#[macro_use]
extern crate derive_new;

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;
