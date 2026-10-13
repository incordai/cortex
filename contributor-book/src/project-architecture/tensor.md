# Tensor

The public tensor type is `Tensor<const D: usize, K = Float>`: `D` is its rank and `K` is `Float`,
`Int`, or `Bool`. Backend and element precision are runtime properties, selected through `Device`
and `DType`. Models and tensor functions do not have a backend generic parameter.

## From backend generics to runtime selection

Previously, `Tensor<B, D, K>` carried the backend in its Rust type. Model fields and functions
propagated `B: Backend`, and selecting another backend instantiated that generic code for another
backend type. The current API keeps rank and kind in the type while moving backend selection to
runtime values. The same `Tensor<2>` type can represent a tensor on Flex, CUDA, or WGPU.

Cargo features determine which backends are compiled into an application. A `Device` selects the
compute resource and supplies creation settings; existing tensors carry the information needed to
route subsequent operations. Multiple enabled backends can be used side by side, with the same model
code targeting each of them. There is no single process-wide backend that every tensor must use.
Combining tensors still requires compatible devices; transfers are explicit.

The `Backend` trait remains the implementation contract below dispatch. Concrete backends and
decorators such as `Autodiff<B, C>` and `Fusion<B>` still compose using backend types. The change
moves those types out of ordinary application signatures and into the implementation layers.

## Tensor Operations

Operations follow this path:

- `base` => All tensor kinds should implement these operations (reshape, into_data, etc.). The
  implementation is in
  [crates/burn-tensor/src/tensor/api/base.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/base.rs).
- `numeric` => All tensors that are numeric by nature should implement these operations (Add, Sub,
  Div, etc.). The implementation is in
  [crates/burn-tensor/src/tensor/api/numeric.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/numeric.rs).
- `Float` => Tensor operations are only available for float tensors. The implementation is in
  [burn-tensor/src/tensor/api/float.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/float.rs).
- `Int` => Tensor operations are only available for int tensors. The implementation is in
  [burn-tensor/src/tensor/api/int.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/int.rs).
- `bool` => Tensor operations are only available for bool tensors. The implementation is in
  [burn-tensor/src/tensor/api/bool.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/bool.rs).

The layers have distinct responsibilities:

- [`cortex-tensor/src/tensor/api`](https://github.com/tracel-ai/burn/tree/main/crates/burn-tensor/src/tensor/api)
  defines the rank- and kind-checked API, shape checks, and user documentation. `base.rs` covers
  common operations, `numeric.rs` covers numeric kinds, and `float.rs`, `int.rs`, and `bool.rs`
  contain kind-specific operations. Activations and neural-network operations also have function
  APIs in `tensor/activation` and `tensor/module.rs`.
- [`cortex-tensor/src/bridge`](https://github.com/tracel-ai/burn/tree/main/crates/burn-tensor/src/bridge)
  stores tensor primitives opaquely and implements kind-specific forwarding. Thin generic methods
  call non-generic helpers taking `BridgeTensor`, so downstream monomorphization does not repeatedly
  resolve the backend implementation types. Follow the `*_impl` pattern described in the
  `cortex-tensor` crate documentation when adding operations.
- [`cortex-dispatch`](https://github.com/tracel-ai/burn/tree/main/crates/burn-dispatch/src) selects a
  backend from runtime tensors and devices. It also routes autodiff and checkpointing contexts.
  Built-in forwarding uses `#[backend_dispatch]`; extensions use `#[backend_extension]`.
- [`cortex-backend`](https://github.com/tracel-ai/burn/tree/main/crates/burn-backend/src/backend)
  defines `BackendTypes`, `Backend`, and operation traits implemented by concrete backends and
  decorators. These low-level APIs still use backend generics and tensor primitive aliases.

A new kernel operation may need changes at every layer. An operation expressed entirely in terms of
existing tensor operations may need only a public composition. See
[Adding a New Operation](../guides/adding-a-new-operation-to-cortex.md).

`module` => These should be exported as functions instead of methods on tensors. The implementation
is in
[crates/burn-tensor/src/tensor/ops/module.rs](https://github.com/tracel-ai/burn/tree/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/ops/modules).
`activation` => These should also be exported as functions instead of methods on tensors. The
implementation is in
[crates/burn-tensor/src/tensor/ops/activation.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/ops/activation.rs).
Note that some activations are just a combination of backend operations and are not declared in
there.
