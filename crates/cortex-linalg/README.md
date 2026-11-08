# Cortex Linear Algebra

> Linear algebra operations for [Cortex](https://github.com/tracel-ai/burn) tensors

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-linalg.svg)](https://crates.io/crates/cortex-linalg)
[![Documentation](https://docs.rs/cortex-linalg/badge.svg)](https://docs.rs/cortex-linalg)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- Decompositions: `lu`, `qr` and `svd`.
- Linear systems: `solve` for square, nonsingular matrices with vector or matrix right-hand sides.
- `det`, `trace`, `diag`, `outer`, `matvec` and `cosine_similarity`.
- Vector and matrix norms: L0, L1, L2, Lp, max and min absolute value, and `vector_normalize`.

## Usage

Enable both `linalg` and a backend feature on `cortex`, and use the operations through
`cortex::linalg`:

```toml
cortex = { version = "0.22", features = ["linalg", "flex"] }
```

No execution backend is enabled by default when depending on this crate directly; select `flex`,
`wgpu` or another backend feature on it. Enabling only `cortex/flex` does not enable this crate.

Linear algebra previously lived at `cortex_tensor::linalg`; that path is removed in 0.22.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
