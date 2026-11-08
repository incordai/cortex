#![cfg_attr(not(feature = "std"), no_std)]

//! # Cortex Store
//!
//! Advanced model storage and serialization infrastructure for the Cortex deep learning framework.
//!
//! This crate provides comprehensive functionality for storing and loading Cortex modules
//! and their tensor data, with support for cross-framework interoperability, flexible filtering,
//! and efficient memory management through lazy materialization.
//!
//! ## Key Features
//!
//! - **Cortexpack Format**: Native Cortex format with CBOR metadata, ParamId persistence for stateful training, and no-std support
//! - **SafeTensors Format**: Industry-standard format for secure and efficient tensor serialization
//! - **PyTorch Compatibility**: Load PyTorch models directly into Cortex with automatic weight transformation
//! - **Zero-Copy Loading**: Memory-mapped files and lazy tensor materialization for optimal performance
//! - **Flexible Filtering**: Load/save specific model subsets using regex, exact paths, or custom predicates
//! - **Tensor Remapping**: Rename tensors during load/save operations for framework compatibility
//! - **No-std Support**: Core functionality available in embedded and WASM environments
//!
//! ## Quick Start
//!
//! ### Basic Save and Load
//!
//! ```rust,ignore
//! use cortex_store::{ModuleSnapshot, SafetensorsStore};
//!
//! // Save a model
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! model.save_into(&mut store)?;
//!
//! // Load a model
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! model.load_from(&mut store)?;
//! ```
//!
//! ### Loading PyTorch Models
//!
//! ```rust,ignore
//! use cortex_store::PytorchStore;
//!
//! // Load PyTorch model (automatic weight transformation via PyTorchToCortexAdapter)
//! let mut store = PytorchStore::from_file("pytorch_model.pth")
//!     .with_top_level_key("state_dict")  // Access nested state dict if needed
//!     .allow_partial(true);               // Skip unknown tensors
//!
//! model.load_from(&mut store)?;
//! ```
//!
//! ### Filtering and Remapping
//!
//! ```rust,no_run
//! # use cortex_store::SafetensorsStore;
//! // Save only specific layers with renaming
//! let mut store = SafetensorsStore::from_file("encoder.safetensors")
//!     .with_regex(r"^encoder\..*")                         // Filter: only encoder layers
//!     .with_key_remapping(r"^encoder\.", "transformer.")   // Rename: encoder.X -> transformer.X
//!     .metadata("subset", "encoder_only");
//!
//! // Use store with model.save_into(&mut store)?;
//! ```
//!
//! ## Core Components
//!
//! - [`ModuleSnapshot`]: Extension trait for Cortex modules providing `collect()` and `apply()` methods
//! - [`CortexpackStore`]: Native Cortex format with ParamId persistence for stateful training workflows
//! - [`SafetensorsStore`]: Primary storage implementation supporting the SafeTensors format
//! - [`PytorchStore`]: PyTorch model loader supporting .pth and .pt files
//! - [`PathFilter`]: Flexible filtering system for selective tensor loading/saving
//! - [`KeyRemapper`]: Advanced tensor name remapping with regex patterns
//! - [`ModuleAdapter`]: Framework adapters for cross-framework compatibility
//! - [`StoreTensor`]: A tensor on its way from a store to a module, kept as its source handed it over
//! - [`bridge`]: Conversions between cortex-core tensors and the [`cortex_pack::Tensor`] entries the stores move around
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O and other std-only features (default)
//! - `safetensors`: Enables SafeTensors format support (default)
//! - `pytorch`: Enables loading PyTorch `.pt`/`.pth` files (default)
//! - `memmap`: Implied by `std` and adds nothing on top of it. Loading a SafeTensors file
//!   memory-maps it whenever `std` is on, regardless of this flag.

extern crate alloc;

mod adapter;
mod applier;
mod apply_result;
mod collector;
mod filter;
mod tensor;
mod traits;

pub mod bridge;

pub use adapter::{
    CortexToPyTorchAdapter, ChainAdapter, FloatCastAdapter, HalfPrecisionAdapter, IdentityAdapter,
    ModuleAdapter, ModuleContext, PyTorchToCortexAdapter,
};
pub use applier::Applier;
pub use apply_result::{ApplyError, ApplyResult};
pub use collector::Collector;
pub use filter::PathFilter;
pub use tensor::StoreTensor;
pub use traits::{ModuleSnapshot, ModuleStore};

#[cfg(feature = "std")]
mod keyremapper;
#[cfg(feature = "std")]
pub use keyremapper::{KeyRemapper, map_indices_contiguous, map_indices_contiguous_except};

#[cfg(feature = "pytorch")]
pub mod pytorch;
#[cfg(feature = "pytorch")]
pub use pytorch::{PytorchStore, PytorchStoreError};

/// The PyTorch checkpoint reader crate, re-exported.
///
/// [`PytorchStore`] is built on it, wrapping each of its tensors in a [`StoreTensor`], and
/// [`bridge::from_pytorch`] turns one into a [`cortex_pack::Tensor`]. The crate itself reads
/// a checkpoint without going through cortex-pack at all, and exposes the serde deserializer
/// behind [`load_config`](pytorch::PytorchReader::load_config) as
/// [`pytorch_reader::nested`].
#[cfg(feature = "pytorch")]
pub use pytorch_reader;

#[cfg(feature = "safetensors")]
mod safetensors;
#[cfg(feature = "safetensors")]
pub use safetensors::{SafetensorsStore, SafetensorsStoreError};

mod cortexpack;
pub use cortexpack::CortexpackStore;

/// The cortexpack format crate, re-exported.
///
/// [`cortex_pack::Tensor`] is what [`ModuleSnapshot::collect`] returns, and what
/// [`ModuleSnapshot::apply`] takes once converted into a [`StoreTensor`]. A crate
/// holding some can drive a [`cortex_pack::Writer`] itself instead of going through
/// [`CortexpackStore`], which is what you want when the tensors did not come from a
/// [`Module`](cortex_core::module::Module) (weights read out of an ONNX file during codegen,
/// say). Nothing is materialized by collecting; each tensor is read back only when the writer
/// reaches it:
///
/// ```
/// use cortex_store::cortex_pack::{Bytes, Error, Tensor, Writer};
///
/// fn pack(tensors: Vec<Tensor>) -> Result<Bytes, Error> {
///     Writer::new(tensors).into_bytes()
/// }
/// ```
///
/// With cortex-pack's `std` feature, `Writer::write_to_file` is the better choice for a large
/// model: it streams to disk instead of building the container in memory. This example uses
/// [`into_bytes`](cortex_pack::Writer::into_bytes) so it compiles in no-std builds too.
///
/// Reach for this rather than depending on `cortex-pack` directly, so the version always matches
/// the one cortex-store was built against.
pub use cortex_pack;
