use std::sync::Arc;

use crate::search::BlockOptimization;

use super::{ExecutionPlanIndex, InsertQuery, SearchQuery};
use cortex_ir::OperationIr;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default)]
pub struct ExecutionPlanStore<O> {
    plans: Vec<ExecutionPlan<O>>,
    index: ExecutionPlanIndex,
}

/// How a list of operations should be executed.
#[derive(PartialEq, Debug, Clone)]
pub enum ExecutionStrategy<O> {
    /// An optimization was found, and therefore should be executed.
    Optimization {
        opt: O,
        ordering: Arc<Vec<usize>>,
        score: u64,
    },
    /// No optimization was found, each operation should be executed individually.
    Operations { ordering: Arc<Vec<usize>> },
    /// A composition of multiple execution strategies.
    Composed(Vec<Box<Self>>),
}

impl<O> ExecutionStrategy<O> {
    /// The largest operation index any of this strategy's orderings names.
    ///
    /// A plan is cached and matched against a stream it did not run on, so the
    /// indices it carries are a claim about that stream, not a fact. Checking
    /// the claim once, before the walk begins, is what keeps a plan that does
    /// not fit from indexing past the end of the segment part way through —
    /// where the panic would be raised outside every scope, with nothing able
    /// to say which tensors it left unwritten.
    pub fn max_index(&self) -> Option<usize> {
        match self {
            Self::Optimization { ordering, .. } | Self::Operations { ordering } => {
                ordering.iter().copied().max()
            }
            Self::Composed(items) => items.iter().filter_map(|item| item.max_index()).max(),
        }
    }
}

impl<O: crate::NumOperations> ExecutionStrategy<O> {
    pub fn max_relative_shape_id(&self) -> Option<usize> {
        match self {
            Self::Optimization { opt, .. } => opt.max_relative_shape_id(),
            Self::Operations { .. } => None,
            Self::Composed(items) => items
                .iter()
                .filter_map(|item| item.max_relative_shape_id())
                .max(),
        }
    }
}

/// The trigger that indicates when to stop exploring.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum ExecutionTrigger {
    OnOperations(Vec<OperationIr>),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub type ExecutionPlanId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Debug)]
pub struct ExecutionPlan<O> {
    /// The operations on which the exploration is related to.
    pub operations: Vec<OperationIr>,
    /// The criteria that signal when this plan should be executed. Only one trigger is necessary.
    pub triggers: Vec<ExecutionTrigger>,
    /// The optimization that should be used when executing this plan.
    pub optimization: BlockOptimization<O>,
}

impl<O: core::fmt::Debug> ExecutionPlanStore<O> {
    pub fn new() -> Self {
        Self {
            plans: Vec::new(),
            index: ExecutionPlanIndex::default(),
        }
    }

    pub fn find(&self, query: SearchQuery<'_>) -> Vec<ExecutionPlanId> {
        self.index.find(query)
    }

    pub fn add(&mut self, exploration: ExecutionPlan<O>) -> ExecutionPlanId {
        if exploration.operations.is_empty() {
            panic!("Can't add an empty optimization.");
        }

        let id = self.plans.len();

        self.index.insert(InsertQuery::NewPlan {
            operations: &exploration.operations,
            id,
        });

        self.plans.push(exploration);

        id
    }

    pub fn get_mut_unchecked(&mut self, id: ExecutionPlanId) -> &mut ExecutionPlan<O> {
        &mut self.plans[id]
    }

    pub fn get_unchecked(&self, id: ExecutionPlanId) -> &ExecutionPlan<O> {
        &self.plans[id]
    }

    /// Add a new end condition for an optimization.
    pub fn add_trigger(&mut self, id: ExecutionPlanId, trigger: ExecutionTrigger) {
        let criteria = &mut self.plans[id].triggers;

        if !criteria.contains(&trigger) {
            criteria.push(trigger);
        }
    }
}

#[cfg(test)]
mod max_index_tests {
    use super::*;

    /// `()` stands in for an optimization: `max_index` reads only the
    /// orderings, so the payload type is irrelevant to what is under test.
    fn ops(positions: &[usize]) -> ExecutionStrategy<()> {
        ExecutionStrategy::Operations {
            ordering: Arc::new(positions.to_vec()),
        }
    }

    /// THE CRASH, AS A UNIT TEST. `[1, 2, 3]` is the exact ordering the device
    /// runner died on. Three entries, highest index 3 — so it needs FOUR
    /// operations, and every guard that compared a COUNT read 3 and admitted a
    /// queue of 3.
    #[test]
    fn an_ordering_of_stream_positions_reaches_past_its_own_length() {
        let positions = [1usize, 2, 3];
        let max = ops(&positions).max_index().expect("non-empty ordering");
        assert_eq!(max, 3, "[1,2,3] reaches index 3");
        // THE DEFECT, STATED AS A RELATION RATHER THAN ASSERTED AS A CONSTANT:
        // a queue sized by the count cannot hold the highest index. If these
        // two ever coincide for this input the bound is no longer
        // distinguishable from the count it replaced.
        assert!(
            max >= positions.len(),
            "a count of {} would admit a queue that cannot hold index {max}",
            positions.len(),
        );
    }

    /// Contiguous-from-zero is the case where count and reach agree, which is
    /// why the count-based guard looked right for as long as it did.
    #[test]
    fn a_zero_based_ordering_is_the_case_where_a_count_happens_to_work() {
        assert_eq!(ops(&[0, 1, 2]).max_index(), Some(2));
    }

    /// Order within the ordering must not change the answer — the reach is the
    /// maximum, not the last element.
    #[test]
    fn the_reach_is_the_maximum_not_the_final_entry() {
        assert_eq!(ops(&[3, 0, 2, 1]).max_index(), Some(3));
    }

    /// A composition indexes ONE operation list, so its reach is the furthest
    /// any member gets — not the sum of their lengths, which would demand a
    /// queue larger than anything the plan touches and refuse valid plans.
    #[test]
    fn a_composition_reaches_as_far_as_its_furthest_member() {
        let c = ExecutionStrategy::Composed(vec![
            Box::new(ops(&[0, 1])),
            Box::new(ops(&[5])),
            Box::new(ops(&[2, 3])),
        ]);
        assert_eq!(c.max_index(), Some(5), "member [5] reaches index 5");
    }

    /// An empty plan names no index at all.
    #[test]
    fn an_empty_ordering_names_no_index() {
        assert_eq!(ops(&[]).max_index(), None);
        assert_eq!(ExecutionStrategy::<()>::Composed(vec![]).max_index(), None);
    }
}
