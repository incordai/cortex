# Adding a New Operation to Cortex

## Choosing where the operation belongs

First consider the operation's intended users and scope:

`cortex-tensor` is the crate that defines all tensor operations that need to be implemented by the
various backends. The core of this lies in
[crates/burn-backend/src/tensor/ops/numeric.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend/src/tensor/ops/numeric.rs#L17),
which is home to the numeric trait. The numeric trait is the home of all tensor operations that are
numeric in nature and that are shared by `Int` and `Float` Tensor types. The numeric trait is
implemented in
[crates/burn-backend/src/tensor/ops/int.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend/src/tensor/ops/int.rs)
for the int type and in
[crates/burn-backend/src/tensor/ops/float.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend/src/tensor/ops/float.rs)
for the float type. More information on the relationship between Tensor modules can be found under
the section for [Tensor Architecture](../project-architecture/tensor.md#tensor-operations).

Then decide how to implement the operation. A composition of existing tensor operations can be
exposed as a function or extension trait in any of these locations without changing the backend
contract. If custom kernels are needed, a
[backend extension](https://burn.dev/books/burn/advanced/backend-extension/) lets you define the
operation and its backend implementations in an extension crate, including one maintained outside
Cortex.

The sections below describe adding an operation to the core tensor API and, when a new primitive is
needed, its backend contract and routing. For domain or external extensions, follow the backend
extension guide and the conventions of the crate that owns the operation.

## Public tensor API and bridge

In this case, we don't need to worry about `Bool` Tensors. `Float` ops are implemented under
[crates/burn-backend/src/backend/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend/src/backend/ops/tensor.rs),
and `Int` ops under
[crates/burn-backend/src/backend/ops/int_tensor.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend/src/backend/ops/int_tensor.rs).
The current convention is ops of each type, if not unique to that type, are prefixed with the type.
So `powf` and sundry would be defined as `int_powf` for `IntTensorOps` and `float_powf` for
`FloatTensorOps`. If an op is unique to a type, then it should be implemented under
`cortex-tensor/src/api/{type}.rs`. For example, here is an implementation for
[`sin` under `crates/burn-tensor/src/api/float.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/float.rs#L82)
which obviously doesn't make sense for `Int` or `Bool` tensors.

The public type is `Tensor<D, K>`, with an opaque `BridgeTensor` primitive. Keep generic method
bodies thin: route through a non-generic `*_impl` helper where needed and the corresponding kind
operations in `crates/cortex-tensor/src/bridge/ops`. Do not expose backend types in ordinary public
method signatures. See [Tensor Architecture](../project-architecture/tensor.md).

## Backend contract and dispatch

Define the primitive operation in the relevant trait under `crates/cortex-backend/src/backend/ops`.
Shared names are prefixed by kind, such as `float_powf` and `int_powf`. A default implementation may
compose existing primitive operations where appropriate. Dtypes are runtime values; there are no
backend-associated float or integer element types.

Additional tests should be added to `cortex-backend-tests` under
[`crates/burn-backend-tests/tests/tensor/{float_or_int}/ops/{op_name}.rs`](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend-tests/tests/tensor/float/ops/powf.rs),
and the module name should be inserted into
`crates/cortex-backend-tests/tests/tensor/{float_or_int}/ops/mod.rs`.

If it makes sense for a floating point operation to support quantization, the
[`QTensorOps`](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend/src/backend/ops/qtensor.rs#L117)
counterpart is usually added at the same time with a default implementation (as mentioned in the
previous section). Tests for `q_*` ops follow a similar procedure: the test is added under
[`crates/burn-backend-tests/tests/tensor/float/quantization/ops/extended/{op_name}.rs`](https://github.com/tracel-ai/burn/tree/9f31281/crates/burn-backend-tests/tests/tensor/float/quantization/ops/extended),
the module name is inserted into
[`crates/burn-backend-tests/tests/tensor/float/quantization/ops/extended/mod.rs`](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-backend-tests/tests/tensor/float/quantization/ops/extended/mod.rs).
If you take a look at any of the existing tests for an operation on a quantized tensor, you will see
that the inputs and expected outputs are always defined with floating point values. While it assumes
that the quantization and dequantization are correct, it makes the tests much more readable and
easier to understand w.r.t. what is being tested. Effectively, the tests are there to ensure that a
tensor operation is invariant to quantization (up to some quantization error, of course).

## Concrete backends and decorators

Implement the operation on the supported concrete backends. For CubeCL kernels, the implementation
is on `CubeBackend`; runtime-specific execution is selected by its device. Handle supported dtypes
and layouts, including non-contiguous inputs where the operation permits them.

Since this is probably the hardest and the least straightforward, we'll cover this backend
separately. `cortex-autodiff` enables other backends to use autodifferentiation[^autodiff]. Ops for
float types are implemented in
[crates/burn-autodiff/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-autodiff/src/ops/tensor.rs)
and need to:

- **Autodiff:** implement the derivative in `crates/cortex-autodiff/src/ops`. Follow neighboring
  operations for `Backward`, saved state, checkpointing, broadcasting reductions, and tracked versus
  untracked inputs. The forward pass must save only the state needed by the backward computation.
- **IR:** add the representation under `crates/cortex-ir/src/operation.rs` when the operation is
  recorded or transmitted, including shape and scalar arguments.
- **Fusion:** record the operation in `crates/cortex-fusion/src/ops`. Kernel fusion support, when
  appropriate, also involves `cortex-cubecl-fusion`; merely recording an operation does not make it
  fusible with its neighbors.
- **Router:** record the operation in `crates/cortex-router/src/ops` and add its execution to
  `TensorInterpreter` in `crates/cortex-router/src/interpreter.rs`. The recorded IR and interpreter
  must agree on the operation's inputs, outputs, and metadata.
- **Remote and capture:** verify the operation through these consumers of the router layer.
  `cortex-remote` uses `BackendRouter<RemoteChannel>` and executes received operations through
  `TensorInterpreter`; `cortex-capture` uses `BackendRouter<CaptureChannel>` to record them without
  execution. Ordinary operation support belongs in the shared router layer; changes in these crates
  are needed when the operation requires additional transport or capture handling.

Some operations have intentional backend limitations. Make them explicit in documentation and errors
rather than assuming every runtime has the same capability.

## Tests and documentation

Add forward and backward coverage to `cortex-backend-tests`; see the
[testing guide](../getting-started/testing.md). Verify shapes, values, broadcasting, and dtypes,
plus empty inputs or non-contiguous layouts when relevant. Check every differentiable input and any
saved state used by the backward pass.

Since `pow` is a binary operation, the left and right functions are the partial derivatives with
respect to the left and right sided tensors.

Let's define the operator as a function \\(f(x,y)=x^{y}\\) , where \\(x\\) is the left hand tensor
and \\(y\\) is the right handed tensor. The two closures are defining the partial derivatives of
\\(f\\) with respect to \\(x\\),\\(y\\). Treat the other variables as a constant

$$\frac{\delta }{\delta x} (x^{y})= y \cdot x^{y-1}$$ is the left handed closure, and

$$\frac{\delta }{\delta y} (x^{y}) = x^{y} \cdot ln(x)$$

is the right. If you aren't sure how to calculate these by hand, it is recommended to use
[symbolab](<https://www.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(x%5E%7By%7D%5Cright)?or=input>),
plug in your operator in terms of \\(x\\) and \\(y\\), and just swap out the variable
\\(x\\)|\\(y\\) in the partial derivative to get the other side.

### Testing autodiff

For testing the `autodiff` operations, please refer to
[this section](../getting-started/testing.md).

## Adding the Op to other backends

Most of these are fairly straightforward implementations. For reference here's pow's float
implementation for torch and flex backends:

1. Torch implementation in
   [crates/burn-tch/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tch/src/ops/tensor.rs#L467)
   and the Op used in
   [crates/burn-tch/src/ops/base.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tch/src/ops/base.rs#L481)
2. Flex in
   [crates/burn-flex/src/ops/float.rs](https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/src/ops/float.rs)

This is where any calculation happens currently. Playing a guessing game with method names and
seeing what completions are suggested will take you far. If you are having trouble figuring out how
to do it from the docs for that backend,
[try searching github for relevant function calls](https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax).

## Adding the Op to fusion, JIT and cubecl backends

Adding an operator to these backends can be fairly straightforward, though due to what these
backends are for, involves a bit more indirection. Fusion and jit, like autodiff, are not target
backends as much as backends that enable certain functionality for other backends, in this case
kernel fusion or just-in-time compilation. Adding the operator won't involve doing any calculation,
you'll just be describing how the generated code should look. Most of this can be
copy/pasted/adjusted from other functions.

Here's how powf was added to `cortex-fusion`:

1. Added powf to the float ops under
   [crates/burn-fusion/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-fusion/src/ops/tensor.rs#L2061)
2. Added powf to the `NumericOperationIr` enum under
   [crates/burn-ir/src/operation.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-ir/src/operation.rs#L564)
3. Added powf to the implementations of `NumericOperationIr` enum under
   [crates/burn-ir/src/operation.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-ir/src/operation.rs#L1086)
4. Added powf to the implemented of `NumericOperationIr` enum under
   [burn/crates/burn-fusion/src/stream/context.rs](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-fusion/src/stream/context.rs#L883)

The way `cubecl` handles tensor-scalar operations is by transforming both into a sequence of
vectorized scalar operations. Since powf already existed in `cubecl`, it was pretty easy to reuse
the existing implementation for the situation where both sides of the operation were tensors. The
`cubecl` crate is primarily concerned with how the operation is compiled and executed by the gpu.
The actual implementation is defined in `cortex-cubecl`.

Here is where code was added for powf in `cortex-cubecl` and `cubecl`:

1. to the implementation of
   [`FloatTensorOps` under `burn/crates/burn-cubecl/src/ops/tensor.rs`](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-cubecl/src/ops/tensor.rs#L578)
2. the function being called was added to
   [`burn/crates/burn-cubecl/src/ops/numeric.rs`](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-cubecl/src/ops/numeric.rs#L211-L214)
3. the operator was defined in
   [`cubecl/crates/cubecl-ir/src/arithmetic.rs`](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-ir/src/arithmetic.rs#L41)
4. how the operation looks to the gpu was added to
   [`burn/crates/burn-cubecl-fusion/src/engine/codegen/ir.rs`](https://github.com/tracel-ai/burn/blob/9f31281/crates/burn-cubecl-fusion/src/engine/codegen/ir.rs#L97)
5. the mappings between the gpu operation and the CPP, WGSL and SPIR-V instructions were added to
   [`cubecl/crates/cubecl-cpp/src/shared/base.rs`](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-cpp/src/shared/base.rs#L1285),
   [`cubecl/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs`](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L869)
   and
   [`cubecl/crates/cubecl-spirv/src/arithmetic.rs`](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-spirv/src/arithmetic.rs#L491)
6. the instructions themselves were added for WGSL to
   [instruction op enum in `cubecl/crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs#L124),
   and the actual
   [instruction in wgsl here](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs#L654),
   for CPP in the enum here
   [`cubecl/crates/cubecl-cpp/src/shared/instruction.rs`](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-cpp/src/shared/instruction.rs#L187)
   and the actual instruction here
   [`cubecl/crates/cubecl-cpp/src/shared/binary.rs`](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-cpp/src/shared/binary.rs#L216)

We needed to generate some custom WGSL code for powf in WGSL, primarily due to issues with proper
case handling of the wgsl pow function, like 0 to the 0 power being 1, and any negative number to an
even power being positive. We reused as much as the existing logic as possible, and then branched at
the last point based off the var type of the rhs.
[See here](https://github.com/tracel-ai/cubecl/blob/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L1229).
For most operations, you shouldn't need to add to `cubecl-wgpu/src/compiler/wgsl/extension.rs`
unless the operation isn't native to WGSL.

For functions that need a complex kernel without a direct mapping to a base instruction, simply use
the `cube` macro (see
[the `cubecl` book](https://github.com/tracel-ai/cubecl/tree/88c0c6f781f70ad2f6e9981fd0cbe2e87e153a35/cubecl-book)).

And you're done! Congrats, you just fully added a new operation to cortex, and we are all one step
closer to the answer to [Are we learning yet?](https://www.arewelearningyet.com/) being "Yes, and
it's freaking fast!". Buy yourself a coffee.

[^supertrait]:
    for more on supertraits see
    [the advanced trait section of the rust book](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait)

[^autodiff]:
    wiki link for
    [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

[^absolute_units]:
    for more information on unit structs see
    [the defining and instantiating structs section of the rust book](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#unit-like-structs-without-any-fields)
