#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "tch", allow(deprecated))]
#![cfg_attr(feature = "ndarray", allow(deprecated))]

//! Linear algebra operations for Cortex tensors.
//!
//! No execution backend is enabled by default. Select `flex`, `wgpu`, or another
//! backend feature on this crate. When configuring through `cortex`, enable both
//! `linalg` and the backend feature, for example `features = ["linalg", "flex"]`.
//! Enabling only `cortex/flex` does not enable this crate's backend implementations.
//!
//! # Migration
//!
//! Linear algebra previously lived at `cortex_tensor::linalg` (and therefore
//! `cortex_core::tensor::linalg`). Those direct-crate paths are intentionally
//! removed. Use this crate directly, or `cortex::linalg` from the umbrella crate.

extern crate alloc;

use cortex_core::tensor::{AsIndex, Bool, DType, Device, ElementConversion, Float, Int, Tensor, s};

macro_rules! check {
    ($check:expr) => {
        if let $crate::check::TensorCheck::Failed(check) = $check {
            core::panic!("{}", check.format());
        }
    };
}

mod check;
mod functions;
mod ops;
mod svd_host;

// Keep the moved implementations focused on their algorithms. These private
// aliases mirror the old cortex-tensor layout without restoring its public API.
mod kind {
    pub(crate) use cortex_core::tensor::kind::{Basic, Numeric, Ordered};
}
mod tensor {
    pub(crate) use cortex_core::tensor::{Int, Shape, Tensor};
}
pub use functions::*;
pub use ops::LinalgOps;
