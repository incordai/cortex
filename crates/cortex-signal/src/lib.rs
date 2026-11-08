#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Signal processing operations for Cortex tensors.
//!
//! No execution backend is enabled by default. Select `flex`, `wgpu`, or another
//! backend feature on this crate. When configuring through `cortex`, enable both
//! `signal` and the backend feature, for example `features = ["signal", "flex"]`.
//! Enabling only `cortex/flex` does not enable this crate's FFT backend implementation.
//!
//! # Migration
//!
//! Signal processing previously lived at `cortex_tensor::signal` (and therefore
//! `cortex_core::tensor::signal`). Use this crate directly or enable Cortex's `signal`
//! feature for `cortex::signal` and the compatibility path `cortex::tensor::signal`.
//! FFTs are backend extensions; windows and STFT/ISTFT compose
//! tensor operations. Enable `autodiff` to differentiate through FFTs.
//!
//! FFT kernels are available for Flex and CubeCL backends.
//! The backend features enable native FFT kernels through `cortex-flex/fft` or
//! `cortex-cubecl/fft`; these kernels are excluded from the backends' default builds.
//!
//! # Remote execution and capture
//!
//! FFTs are recorded as custom operations named `signal::rfft` and `signal::irfft`.
//! Remote servers and interpreters replaying captured graphs must enable `router`
//! and install `register_fft_ops`
//! in their custom-operation registry before execution. Pass that registry to
//! a remote server's `with_custom_ops` or `TensorInterpreter::with_custom_ops`.
//! Enable `remote` on the client and the concrete backend feature on the server.

extern crate alloc;

mod backends;
#[cfg(any(feature = "fusion", feature = "router"))]
mod custom;
mod functions;
mod ops;

#[cfg(feature = "router")]
pub use backends::router::register_fft_ops;
pub use functions::*;
pub use ops::SignalOps;
