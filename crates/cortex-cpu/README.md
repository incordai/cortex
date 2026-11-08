# Cortex CPU Backend

[Cortex](https://github.com/tracel-ai/burn) CubeCL CPU backend

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-cpu.svg)](https://crates.io/crates/cortex-cpu)
[![Documentation](https://docs.rs/cortex-cpu/badge.svg)](https://docs.rs/cortex-cpu)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

This crate provides a CPU backend for [Cortex](https://github.com/tracel-ai/burn) using
[CubeCL](https://github.com/tracel-ai/cubecl.git)'s CPU runtime. It JIT-compiles shared CubeCL
kernels to native CPU code through LLVM, with kernel fusion and autotuning enabled by default. LLVM
is bundled automatically, no system installation is required.

## cortex-cpu vs cortex-flex

Cortex has two independent CPU backends. Neither one is built on top of the other.

|                    | `cortex-cpu`                                   | [`cortex-flex`](https://github.com/tracel-ai/burn/tree/main/crates/burn-flex)                   |
| ------------------ | -------------------------------------------- | --------------------------------------------- |
| Implementation     | CubeCL kernels compiled through LLVM         | Hand-written Rust kernels, `gemm`, SIMD       |
| Execution          | JIT-compiled, with fusion and autotune       | Eager                                         |
| Native deps        | Bundled LLVM, no system install              | None (pure Rust)                              |
| `no_std` / Wasm    | No                                           | Yes                                           |
| `cortex` feature     | `cpu`                                        | `flex`                                        |
| Device constructor | `Device::cpu()`                              | `Device::flex()`                              |

Use `cortex-cpu` when you want the CubeCL stack (fusion, custom CubeCL kernels shared with the GPU
backends) on the CPU. Use `cortex-flex` for a lightweight, portable CPU backend that also runs on
`no_std` and WebAssembly targets.

## Usage Example

```toml
cortex = { version = "0.22", features = ["cpu"] }
```

```rust, ignore
use cortex::tensor::{Device, Tensor};

let device = Device::cpu();
let tensor = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
```

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
