# Cortex CubeCL Backend

> The [Cortex](https://github.com/tracel-ai/burn) backend for every [CubeCL](https://github.com/tracel-ai/cubecl) runtime

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-cubecl.svg)](https://crates.io/crates/cortex-cubecl)
[![Documentation](https://docs.rs/cortex-cubecl/badge.svg)](https://docs.rs/cortex-cubecl)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`CubeBackend` implements Cortex's tensor operations as CubeCL kernels, compiled just in time for the
device they run on. CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu and the CubeCL CPU runtime all share
this one backend type; a tensor's device says which runtime it uses.

## Usage

Applications select a runtime with a Cortex feature and a `Device` constructor:

| Runtime | Cortex feature | Device                               | Crate                     |
| ------- | ------------ | ------------------------------------ | ------------------------- |
| CUDA    | `cuda`       | `Device::cuda(0)`                    | [cortex-cuda](https://github.com/tracel-ai/burn/tree/main/crates/burn-cuda) |
| ROCm    | `rocm`       | `Device::rocm(0)`                    | [cortex-rocm](https://github.com/tracel-ai/burn/tree/main/crates/burn-rocm) |
| wgpu    | `wgpu`       | `Device::wgpu(Default::default())`   | [cortex-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| Metal   | `metal`      | `Device::metal(Default::default())`  | [cortex-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| Vulkan  | `vulkan`     | `Device::vulkan(Default::default())` | [cortex-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| WebGPU  | `webgpu`     | `Device::webgpu(Default::default())` | [cortex-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| CPU     | `cpu`        | `Device::cpu()`                      | [cortex-cpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-cpu)   |

Use this crate directly to write custom kernels: `kernel` and `ops` hold the building blocks, and
`cubecl` is re-exported so kernels use the same CubeCL version as the backend. See the
[custom CubeCL kernel](https://cortex.dev/books/cortex/advanced/backend-extension/custom-cubecl-kernel.html)
chapter of the Cortex Book.

## Feature Flags

- `cuda`, `hip`, `wgpu`, `metal`, `vulkan`, `webgpu`, `cpu`: compile in a CubeCL runtime.
- `fusion` (default): kernel fusion through [cortex-fusion](https://github.com/tracel-ai/burn/tree/main/crates/burn-fusion).
- `autotune` (default): benchmark kernel variants at runtime and keep the fastest.
- `fft`: FFT kernels.
- `template`: launch hand-written, non-JIT kernels.
- `tracing`: instrument operations with the `tracing` crate.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
