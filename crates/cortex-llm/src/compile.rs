//! `compile`: run a tensor function as a recorded device graph, one graph
//! per input shape, without the caller managing buffers.
//!
//! Built on cortex's `capture` / `Graph::replay` (cortex-tensor
//! `tensor/api/graph.rs`), which records a closure's kernel launches once and
//! replays them as a single dispatch on backends with hardware graphs
//! (CUDA/HIP) — but which replays against the exact buffers of the capture,
//! leaving the caller to refresh inputs in place and keep everything alive
//! (hence its `unsafe`). [`Compiled`] owns that contract:
//!
//! - it keeps its own input buffers per input signature (shapes and dtypes)
//!   and writes each call's inputs into them in place, checking first that
//!   it is the buffers' only owner (`Tensor::can_mut`) so the write really
//!   lands in the recorded buffer;
//! - the graph's closure holds those buffers, so they live as long as the
//!   graph;
//! - outputs are returned as copies, so a later call never changes a tensor
//!   the caller already holds;
//! - whenever the in-place refresh cannot be guaranteed, or the backend has
//!   no hardware graphs, the function simply runs eagerly — slower, never
//!   wrong. On backends without graphs, kernel fusion (cortex-fusion) still
//!   applies to every call.
//!
//! The function must be pure: the same inputs give the same output, and it
//! reads no tensor other than its arguments and the state it owns.
//!
//! Callers: model code (a decode step is the canonical use). Calls:
//! `cortex_core::tensor::capture`. Not `Send`: a compiled function lives on
//! the thread (and device stream) that created it, as the graph requires.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use std::collections::HashMap;

use cortex_core::tensor::{DType, Device, Graph, Tensor, capture};

type Signature = Vec<(Vec<usize>, DType)>;
type Inputs<const DI: usize> = Rc<RefCell<Vec<Tensor<DI>>>>;
type Step<const DO: usize> = alloc::boxed::Box<dyn FnMut() -> Tensor<DO>>;

struct Entry<const DI: usize, const DO: usize> {
    inputs: Inputs<DI>,
    graph: Graph<Tensor<DO>, Step<DO>>,
}

/// A function of `DI`-dimensional tensors returning a `DO`-dimensional one,
/// replayed from a recorded graph per input signature.
pub struct Compiled<const DI: usize, const DO: usize, F> {
    device: Device,
    function: Rc<RefCell<F>>,
    graphs: HashMap<Signature, Entry<DI, DO>>,
    replays: usize,
}

/// Compiles `function` for `device` (see the module docs).
pub fn compile<const DI: usize, const DO: usize, F>(device: &Device, function: F) -> Compiled<DI, DO, F>
where
    F: FnMut(&[Tensor<DI>]) -> Tensor<DO> + 'static,
{
    Compiled {
        device: device.clone(),
        function: Rc::new(RefCell::new(function)),
        graphs: HashMap::new(),
        replays: 0,
    }
}

/// A copy of `t` in a fresh buffer.
fn fresh<const D: usize>(t: &Tensor<D>) -> Tensor<D> {
    Tensor::zeros(t.dims(), (&t.device(), t.dtype())) + t.clone()
}

impl<const DI: usize, const DO: usize, F> Compiled<DI, DO, F>
where
    F: FnMut(&[Tensor<DI>]) -> Tensor<DO> + 'static,
{
    /// `function(inputs)`, from the recorded graph of this input signature
    /// (recording it on first use).
    pub fn call(&mut self, inputs: &[Tensor<DI>]) -> Tensor<DO> {
        if inputs.iter().any(|t| t.device() != self.device) {
            return (self.function.borrow_mut())(inputs);
        }
        let signature: Signature = inputs.iter().map(|t| (t.dims().to_vec(), t.dtype())).collect();

        if !self.graphs.contains_key(&signature) {
            let owned: Inputs<DI> = Rc::new(RefCell::new(inputs.iter().map(fresh).collect()));
            let (held, function) = (owned.clone(), self.function.clone());
            let step: Step<DO> = alloc::boxed::Box::new(move || {
                let args = held.borrow().clone();
                (function.borrow_mut())(&args)
            });
            let graph = capture(&self.device, step);
            let first = fresh(graph.output());
            self.graphs.insert(signature, Entry { inputs: owned, graph });
            return first;
        }

        let entry = self.graphs.get_mut(&signature).expect("checked above");
        if !entry.graph.is_hardware() {
            // The replay re-runs the closure on whatever the slots hold.
            *entry.inputs.borrow_mut() = inputs.to_vec();
            // Safety: the fallback path re-executes the closure; no recorded
            // buffers are involved.
            let out = unsafe { entry.graph.replay() }.clone();
            entry.inputs.borrow_mut().clear();
            return out;
        }

        // Hardware path: the recorded kernels read the slots' buffers, so the
        // new inputs must be written into exactly those buffers.
        let mut slots = entry.inputs.borrow_mut();
        if slots.iter().any(|s| !s.can_mut()) {
            drop(slots);
            return (self.function.borrow_mut())(inputs);
        }
        for (slot, new) in slots.iter_mut().zip(inputs) {
            let ranges: [core::ops::Range<usize>; DI] = core::array::from_fn(|i| 0..new.dims()[i]);
            slot.inplace(|s| s.slice_assign(ranges, new.clone()));
        }
        drop(slots);
        self.replays += 1;
        // Safety: every buffer the graph touches is owned by its closure (the
        // input slots, the function's state) or by the graph (the output);
        // the slots were refreshed in place on this thread's stream just
        // above; `Compiled` is not `Send`, so no other thread or stream can
        // use them; and the output is copied before it is handed out.
        fresh(unsafe { entry.graph.replay() })
    }

    /// Input signatures recorded so far.
    pub fn signatures(&self) -> usize {
        self.graphs.len()
    }

    /// Calls served by a hardware graph replay (zero on backends without
    /// hardware graphs).
    pub fn hardware_replays(&self) -> usize {
        self.replays
    }
}
