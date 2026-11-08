# Cortex Fusion

> Kernel fusion for [Cortex](https://github.com/tracel-ai/burn), as a backend decorator

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-fusion.svg)](https://crates.io/crates/cortex-fusion)
[![Documentation](https://docs.rs/cortex-fusion/badge.svg)](https://docs.rs/cortex-fusion)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`Fusion<B>` defers a backend's operations instead of running each one immediately, then groups
them into optimizations. Several element-wise operations can run as one kernel, and the
intermediate tensors between them never reach device memory.

The CubeCL backends (CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu and the CubeCL CPU runtime) enable
fusion by default, so applications get it without naming this crate. The fused kernels themselves
live in [`cortex-cubecl-fusion`](https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl-fusion).

Backend authors opt in by implementing `FusionBackend` and `FusionRuntime`. Backend extensions can
register their own fused operations through the `custom` module.

## Feature Flags

- `std` (default): standard library support.
- `memory-checks`: check the fusion runtime for leaked tensors (for tests).
- `tracing`: instrument operations with the `tracing` crate.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
