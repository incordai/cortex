# Cortex CPU Backend

[Burn](https://github.com/tracel-ai/burn) CubeCL CPU backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-cpu.svg)](https://crates.io/crates/burn-cpu)
[![Documentation](https://docs.rs/burn-cpu/badge.svg)](https://docs.rs/burn-cpu)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

This crate provides a MLIR based CPU backend for [Burn](https://github.com/tracel-ai/burn) using the
[cubecl](https://github.com/tracel-ai/cubecl.git) crates.

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
[Cortex Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
