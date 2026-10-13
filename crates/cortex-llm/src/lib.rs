#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! LLM inference building blocks for Cortex tensors.
//!
//! - [`quant`]: the packed quantized weight format (2-8 bit codes, grouped
//!   scales, symmetric or affine) and its host-side reference.
//! - [`QuantizedMatrix`] / [`QuantizedExperts`]: packed weights on a device
//!   and the (gathered) matmuls that read them without unpacking to full
//!   precision on CubeCL backends.
//! - [`QLoraLinear`]: a frozen packed base with a trainable low-rank adapter;
//!   with the `autodiff` feature every op here records a backward pass.
//!
//! No execution backend is enabled by default. Select `flex`, `wgpu`, `cuda`
//! or another backend feature on this crate.

extern crate alloc;

#[cfg(feature = "autodiff")]
mod autodiff;
pub mod cache;
#[cfg(feature = "std")]
mod compile;
#[cfg(feature = "std")]
pub mod distributed;
#[cfg(feature = "std")]
pub mod checkpoint;
mod fused;
#[cfg(feature = "fusion")]
mod fusion;
pub mod generate;
#[cfg(feature = "cubecl-backend")]
mod kernels;
#[cfg(feature = "std")]
pub mod llama;
mod ops;
mod parallel;
mod placement;
pub mod prefix;
mod qcache;
mod qlora;
pub mod quant;
mod quantized;
pub mod sampler;

pub use cache::{KvCache, RotatingKvCache, cache_attention};
#[cfg(feature = "std")]
pub use compile::{Compiled, compile};
pub use fused::{rms_norm, rms_norm_reference, rope, rope_inv_freq, rope_reference};
pub use generate::{GenerateOptions, LanguageModel, StopReason, generate, speculative_generate};
#[cfg(feature = "std")]
pub use llama::{Llama, LlamaConfig, LlamaState};
pub use prefix::{PrefixHit, PrefixState, PromptCache};
pub use sampler::{Distribution, Sampler, SamplerConfig};
pub use ops::LlmOps;
pub use parallel::{ColumnParallelLinear, RowParallelLinear};
pub use placement::LayerPlacement;
pub use qcache::QuantizedKvCache;
pub use qlora::QLoraLinear;
pub use quantized::{QuantizedExperts, QuantizedMatrix};
