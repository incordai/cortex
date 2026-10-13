# Cortex Core

This crate should be used with [burn](https://github.com/tracel-ai/burn). It contains the core
traits and components for building and training deep learning models with Cortex.

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-core.svg)](https://crates.io/crates/burn-core)
[![Documentation](https://docs.rs/burn-core/badge.svg)](https://docs.rs/burn-core)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Applications should depend on [cortex](https://crates.io/crates/burn), which re-exports everything
here together with neural network layers, optimizers and training. This crate is the dependency for
libraries that only need the core abstractions.

- `tensor`: Cortex's tensor API, re-exported from [cortex-tensor](https://github.com/tracel-ai/burn/tree/main/crates/burn-tensor).
- `module`: the `Module` trait, implemented with `#[derive(Module)]`, and `Param` for trainable
  tensors.
- `config`: serializable configuration structs with `#[derive(Config)]`.
- `store`: module records and the cortexpack format.
- `data`: datasets, batchers and data loaders (`std` only).
- `prelude`: the types most programs import.

## Feature Flags

Backend features (`wgpu`, `cuda`, `flex`, ...) and the `autodiff`, `fusion`, `remote` and `capture`
features match those of `cortex`. Others:

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `dataset`: the dataset library in `data`; `vision`, `audio` and `sqlite` add sources.
- `network`: file downloads with a progress bar.
- `tracing`: instrument operations with the `tracing` crate.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
