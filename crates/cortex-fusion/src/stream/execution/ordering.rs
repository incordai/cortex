use std::sync::Arc;

use cortex_ir::HandleContainer;

use crate::{FusionRuntime, NumOperations, Optimization, UnfusedOp, stream::Context};

/// Manage the execution of potentially multiple optimizations and operations out of order.
pub struct OrderedExecution<R: FusionRuntime> {
    operations: Vec<UnfusedOp<R>>,
    num_executed: usize,
    ordering: Option<Arc<Vec<usize>>>,
}

impl<R: FusionRuntime> OrderedExecution<R> {
    /// Returns the operation that can be executed without impacting the state of the execution.
    ///
    /// This is useful to implement fallback for optimizations.
    #[allow(clippy::borrowed_box)]
    pub fn operation_within_optimization(&self, index: usize) -> UnfusedOp<R> {
        match &self.ordering {
            Some(val) => {
                let index = val[index];
                self.operations[index].clone()
            }
            None => panic!("No ordering provided"),
        }
    }

    pub(crate) fn new(operations: Vec<UnfusedOp<R>>) -> Self {
        Self {
            operations,
            num_executed: 0,
            ordering: None,
        }
    }

    pub(crate) fn finish(mut self) -> (Vec<UnfusedOp<R>>, usize) {
        self.operations.drain(0..self.num_executed);
        (self.operations, self.num_executed)
    }

    pub(crate) fn execute_optimization(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<R::FusionHandle>,
        ordering: Arc<Vec<usize>>,
    ) {
        if ordering.len() > self.operations.len() {
            panic!(
                "Ordering is bigger than operations: ordering len {}, operations len {}, \
                 num_executed {}, optimization len {}, ordering {:?}",
                ordering.len(),
                self.operations.len(),
                self.num_executed,
                optimization.len(),
                ordering,
            );
        }
        self.ordering = Some(ordering);
        let num_drained = optimization.len();
        optimization.execute(context, self);
        self.num_executed += num_drained;
    }

    pub(crate) fn execute_operations(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        ordering: &[usize],
    ) {
        self.num_executed += ordering.len();

        for id in ordering {
            // BOUNDS-CHECKED ON PURPOSE, WITH A DIAGNOSTIC.
            //
            // A bare `self.operations[*id]` panics with nothing but
            // "index out of bounds: the len is 1 but the index is 1", naming
            // neither the stream nor what the ordering asked for. Seen
            // intermittently - roughly one run in five - when two devices are
            // driven concurrently, and always from the fusion device-runner
            // thread, where it surfaces as an opaque
            // `CallError(task panicked on device runner thread)` at
            // client.rs:200, three layers from anything actionable.
            //
            // Note the asymmetry this repairs: `execute_optimization` above
            // already guards, but only that the ordering is no LONGER than the
            // operation list. It never checks the VALUES, and an in-range
            // length can still carry an out-of-range index - exactly the
            // failure observed, where `len` fell to 0 while an index of 1 was
            // still requested.
            //
            // This does NOT fix the root cause: something hands this function
            // an ordering that outlives the operations it indexes. It makes the
            // next occurrence name its own state instead of hiding it.
            // Skipping the operation was rejected deliberately - a silently
            // dropped op yields a wrong tensor, which is worse than a crash.
            let op = self.operations.get(*id).unwrap_or_else(|| {
                panic!(
                    "fusion ordering refers to operation {id}, but only {} operation(s) remain \
                     (num_executed {}, ordering {:?}) - the ordering has outlived the operations \
                     it indexes",
                    self.operations.len(),
                    self.num_executed,
                    ordering,
                )
            });
            op.execute(handles);
        }
    }
}
