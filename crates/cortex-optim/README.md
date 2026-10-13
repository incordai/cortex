# Cortex Optimizers

> Optimizers and learning rate schedulers for [Cortex](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-optim.svg)](https://crates.io/crates/burn-optim)
[![Documentation](https://docs.rs/burn-optim/badge.svg)](https://docs.rs/burn-optim)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Applications use these through `cortex::optim`, `cortex::lr_scheduler` and `cortex::grad_clipping`. An
optimizer is built from its config and applied to a module with gradients from a backward pass:

```rust,ignore
use cortex::optim::{AdamConfig, GradientsParams};

let mut optimizer = AdamConfig::new().init();
let grads = GradientsParams::from_grads(loss.backward(), &model);
model = optimizer.step(learning_rate, model, grads);
```

- Optimizers: SGD, Adam, AdamW, Adagrad, Adafactor, Adan, LAMB, L-BFGS, Lion, Muon and RMSprop,
  with momentum, weight decay and gradient accumulation helpers.
- `lr_scheduler`: constant, step, exponential, linear, cosine, Noam, and sequential or composed
  schedules.
- `grad_clipping`: clipping by value or by norm.

See the [optimizer](https://github.com/tracel-ai/burn/blob/main/burn-book/src/building-blocks/optimizer.md) and
[learning rate scheduler](https://github.com/tracel-ai/burn/blob/main/burn-book/src/building-blocks/lr-scheduler.md) chapters
of the Cortex Book.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `tracing`: instrument operations with the `tracing` crate.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
