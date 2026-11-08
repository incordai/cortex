# cortex-einsum

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-einsum.svg)](https://crates.io/crates/cortex-einsum)
[![Documentation](https://docs.rs/cortex-einsum/badge.svg)](https://docs.rs/cortex-einsum)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Dependency-free equation parsing and execution planning shared by Cortex's `einsum!`
macro and runtime einsum API. The crate supports `no_std` with `alloc`.

Equations support ASCII labels (`A-Z`, then `a-z`), repeated input labels,
ellipsis, scalar operands, and explicit or implicit output. Parsing validates
equation syntax without tensor shapes. `Equation::plan()` determines diagonal
extraction, canonical axis alignment, last-use reductions, and left-to-right
contraction groups and permutations. Ellipsis dimensions remain symbolic so the
same plan supports different operand ranks.

The macro generates tensor operations from this plan at compile time. The
runtime API interprets the same plan. Operand ranks, diagonal sizes, broadcasting,
and shape-dependent matrix multiplication details are checked by the tensor
executor.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
