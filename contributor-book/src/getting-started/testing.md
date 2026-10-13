# Testing

## Tensor operations

Test for tensor operations (generally of the form: given this input, expect it match or approximate
this output) are defined only in
[`crates/burn-tensor/src/test/ops`](https://github.com/tracel-ai/burn/tree/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-tensor/src/tests/ops)
and not in the backends (with the exception of `cortex-autodiff`). The tensor operation tests are
added to the `testgen_all` macro rule in
[`crates/burn-tensor/src/tests/mod.rs`](https://github.com/tracel-ai/burn/blob/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-tensor/src/tests/mod.rs).
This is then propagated to the existing backends without any additional work.

Run the relevant shared suites using the backend aliases defined in
[`crates/cortex-backend-tests/.cargo/config.toml`](https://github.com/tracel-ai/burn/blob/main/crates/burn-backend-tests/.cargo/config.toml).
Start from the repository root and change into the crate directory so Cargo discovers its aliases:

Tests for autodiff go under
[burn-autodiff/src/tests](https://github.com/tracel-ai/burn/tree/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-autodiff/src/tests)
and should verify backward pass correctness. For binary tensor operations, both the left and right
sides need to be verified.

These aliases run in release mode and select the backend features explicitly. Omit `--test` to run
all test targets for that configuration, or append a test-name filter to narrow the run:

```sh
cargo test-flex
cargo test-flex --test tensor matmul
```

Choose the alias for the backend you are changing, such as `cargo test-cuda`, `cargo test-vulkan`,
or `cargo test-metal`. Aliases for backends that support fusion enable it by default; their
`-no-fusion` variants test without fusion. For example, run both configurations when changing CUDA
operations or fusion behavior:

```sh
cargo test-cuda --test tensor
cargo test-cuda-no-fusion --test tensor
```

From the repository root, use `cargo run-checks` for the repository validation workflow. It defaults
to Flex; select another backend with `cargo run-checks --backend <backend>` when working on
backend-specific code.

## Autodiff

Shared backward tests live in
[`crates/cortex-backend-tests/tests/autodiff`](https://github.com/tracel-ai/burn/tree/main/crates/burn-backend-tests/tests/autodiff)
and are registered through `tests/common/autodiff.rs`. Graph engine unit tests also live in
`cortex-autodiff`. For operations with multiple differentiable inputs, verify every input gradient.

Choose small inputs whose derivatives can be calculated independently. Create source leaves on an
autodiff device and call `require_grad()` before the forward pass. Retrieve their gradients from the
result of `backward()`. Check broadcasting and untracked inputs where relevant; a numerically
correct forward pass does not establish a correct backward implementation.

You can also use PyTorch as a
[reference implementation](https://docs.pytorch.org/devlogs/compiler/2026-07-25-pytorch-a-reference-language/)
to obtain expected outputs and gradients for Cortex tests. For example, this small broadcasting case
checks gradients for both operands:

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
y = torch.tensor([5., 6.], requires_grad=True)
output = x * y
output.sum().backward()

print(output.detach().tolist())  # [[5.0, 12.0], [15.0, 24.0]]
print(x.grad.tolist())           # [[5.0, 6.0], [5.0, 6.0]]
print(y.grad.tolist())           # [4.0, 6.0]
```

Use the same inputs, operation parameters, and reduction in the Cortex test, and record these values
as expected data so the test does not depend on PyTorch. Here, the gradient for `y` sums
contributions over the broadcast dimension. For other operations, check that the reference uses
matching semantics and dtypes, and compare with an appropriate tolerance as described below.

## Precision

Shared suites define `FloatElem` and `IntElem` aliases for each test executable and configure the
device defaults before creating tensors. They are not associated types of a `TestBackend`. Use the
aliases in expected data and use `.elem()` when a literal needs conversion.

For approximate floating-point comparisons, follow nearby tests:

```rust,ignore
actual.into_data().assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
```

For integers, use `IntElem` and skip cases whose inputs cannot be represented by the selected dtype.
Exercise additional precision targets when the change depends on dtype or numerical stability.
