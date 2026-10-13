# Module

Modules organize parameters into structures that can be optimized, saved, and loaded.
`#[derive(Module)]` generates parameter traversal and training/validation conversions. A module does
not force the declaration of the forward pass, leaving it up to the implementer to decide how it
should be defined.

Configuration describes a module's structure and hyperparameters; records store its parameters
separately.

## Parameters and traversal

`Param<T>` gives a value an identity and supports lazy initialization. Tensor parameters use that
identity to associate optimizer state and gradients. `Param<Flag>` represents module-owned control
state, such as whether dropout or batch normalization behaves as during training.

`Module::visit` inspects parameters; `Module::map` transforms them. Visitors and mappers have hooks
for float, integer, and boolean tensors, control flags, and module paths. Reparameterizations such
as LoRA have nested parameters that participate in these traversals. `param.base()` reads the stored
base; `param.val()` materializes the effective value, including a reparameterization.

## Training and validation

`Module` includes the `valid(&self)` and `train(self)` transition hooks; the derive generates both
alongside traversal. Training and validation use the same module type, and a module can contain
parameters with different runtime autodiff contexts.

- `valid(&self)` creates a validation snapshot with autodiff and training flags disabled. It keeps
  configured trainability and flag settings, folds reparameterizations into parameter values, and
  removes checkpointing strategies with the autodiff association.
- `train(self)` enables autodiff and applies configured trainability and flags. It does not undo
  explicit freezing, reconstruct folded adapters, or restore discarded checkpointing strategies.
- `no_grad()` persistently disables parameter gradients while leaving control flags unchanged.
- `freeze()` and `unfreeze()` configure both gradients and flags; group variants target subtrees.

Keep the training module and use its `valid()` snapshot for validation. Inspect individual tensors
with `is_autodiff()`, `is_tracked()`, and `is_require_grad()` rather than inferring training state
from the trait or `module.devices()`. Device equality ignores autodiff settings, and the latter
method deduplicates compute resources.

## Optimization

[`Optimizer`](https://github.com/tracel-ai/burn/blob/main/crates/burn-optim/src/optim/module/base.rs)
updates one tensor at a time from its gradient and optional state. `State<D>` implements `Clone` and
`RecordState`, allowing tensors and scalars to be serialized independently of backend types.

[`ModuleOptimizer`](https://github.com/tracel-ai/burn/blob/main/crates/burn-optim/src/optim/module/module_optimizer.rs)
wraps these optimizers and manages module traversal, parameter groups, gradient lookup, device
migration, and per-parameter state. Parameter groups can use different optimizers.

An update proceeds as follows:

1. Run the model and call `loss.backward()`.
2. Convert the tensor gradients with `GradientsParams::from_grads(grads, &model)`.
3. Call `optimizer.step(learning_rate, model, grads)` to obtain the updated module.

In the following, the `Module` trait is defined in
[`crates/burn-core/src/module/base.rs`](https://github.com/tracel-ai/burn/blob/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-core/src/module/base.rs#L83)
and the `Optimizer` trait is defined in
[`crates/burn-core/src/optim/base.rs`](https://github.com/tracel-ai/burn/blob/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-core/src/optim/base.rs#L8)

The solution to this problem comprises multiple parts. Firstly, the `Optimizer` trait is quite
similar to the `Module` trait, in terms of saving and loading the state. Please refer to the
[serialization](./serialization.md) section for more details.

Secondly, two traits were created. The `Optimizer` trait is general and relatively unopinionated,
with a simple `step` method that takes a learning rate, a module, and the gradients. The other
trait, `SimpleOptimizer`, aims to provide an easier API for implementing new optimizers. The goal is
to allow implementations to avoid handling missing gradients, loading and exporting records,
navigating the module parameter structure, handling tracked and untracked tensors, and other such
tasks.

Thirdly, each tensor that will be optimized needs to be wrapped into a `Param` struct, which gives
them an ID used for (de)serialization and to associate the state of the optimizer to each parameter.
The `Module` trait has two ways to navigate over parameters. The first one is the `map` function,
which returns `Self` and makes it easy to implement any transformation and mutate all parameters.
The second one is the `visit` function, which has a similar signature but does not mutate the
parameter tensors.

#### SimpleOptimizer

Located in
[`crates/burn-core/src/optim/simple/base.rs`](https://github.com/tracel-ai/burn/blob/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-core/src/optim/simple/base.rs#L9),
the `SimpleOptimizer` has two major assumptions:

1. The state of the optimizer is linked to each parameter. In other words, each parameter has its
   own optimizer state, decoupled from the other parameters.
2. The state of the optimizer implements `Record`, `Clone`, and has a `'static` lifetime.

The benefits of those assumptions materialize in simplicity with little loss in flexibility. The
state associative type is also generic over the dimension, making it extremely easy to include
tensors in the state that share the same dimensionality as its parameter.

To wrap a simple optimizer into the more general `Optimizer` trait, the `OptimizerAdaptor` struct is
used.

#### OptimizerAdaptor

Located in in
[`crates/burn-core/src/optim/simple/adaptor.rs`](https://github.com/tracel-ai/burn/blob/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-core/src/optim/simple/adaptor.rs#L14),
the `OptimizerAdaptor` is a simple struct composed of a `SimpleOptimizer` and a hashmap with all
records associated with each parameter ID.

When performing an optimization step, the adaptor handles the following:

1. Updates each parameter tensor in the given module using the `Module::map` function.
2. Checks if a gradient for the current tensor exists.
3. Makes sure that the gradient, the tensor, and the optimizer state associated with the current
   parameter are on the same device. The device can be different if the state is loaded from disk to
   restart training.
4. Performs the simple optimizer step using the inner tensor since the operations done by the
   optimizer should not be tracked in the autodiff graph.
5. Updates the state for the current parameter and returns the updated tensor, making sure it's
   properly registered into the autodiff graph if gradients are marked as required.

Note that a parameter can still be updated by another process, as it is the case with running
metrics used in batch norm. These tensors are still wrapped using the `Param` struct so that they
are included in the module's state and given a proper parameter ID, but they are not registered in
the autodiff graph.
