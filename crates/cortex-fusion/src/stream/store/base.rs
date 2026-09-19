use std::sync::Arc;

use crate::search::BlockOptimization;

use super::{ExecutionPlanIndex, InsertQuery, SearchQuery};
use cortex_ir::OperationIr;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default)]
pub(crate) struct ExecutionPlanStore<O> {
    plans: Vec<ExecutionPlan<O>>,
    index: ExecutionPlanIndex,
}

/// How a list of operations should be executed.
#[derive(PartialEq, Debug, Clone)]
pub(crate) enum ExecutionStrategy<O> {
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
    /// How many operations this strategy must be given before its orderings are
    /// safe to index — one past the highest position any of them names.
    ///
    /// A COUNT IS NOT A BOUND, AND CONFUSING THE TWO IS THIS FILE'S RECURRING
    /// BUG. `ordering` holds ABSOLUTE STREAM POSITIONS, not `0..n`:
    /// `unfused_stream_order` builds it straight from a chunk's `positions`, so
    /// a chunk covering positions `[1, 2, 3]` has three entries but reaches
    /// index 3 and therefore needs FOUR operations behind it.
    ///
    /// Two guards have already been written against a count and both let the
    /// crash through. The first compared `AvailableItem::size`; the second
    /// compared `plan.operations.len()`. Each is the number of operations a
    /// plan covers, and neither bounds what its orderings reach — with
    /// `[1, 2, 3]` both read 3 and admit a queue of 3, after which
    /// `execute_operations` indexes position 3 of a three-element list and the
    /// device-runner thread dies with
    ///
    ///   fusion ordering refers to operation 1, but only 0 operation(s) remain
    ///
    /// Returning the reach directly removes the choice: there is now one
    /// quantity, it is the one that governs the indexing, and a caller cannot
    /// pick a cheaper approximation of it by accident.
    pub(crate) fn required_operations(&self) -> usize {
        match self {
            Self::Optimization { ordering, .. } | Self::Operations { ordering } => {
                ordering.iter().copied().max().map_or(0, |m| m + 1)
            }
            // A composition reaches as far as its furthest member: every child
            // indexes the SAME operation list, so the maximum governs rather
            // than the sum.
            Self::Composed(items) => {
                items.iter().map(|i| i.required_operations()).max().unwrap_or(0)
            }
        }
    }
}

/// The trigger that indicates when to stop exploring.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ExecutionTrigger {
    OnOperations(Vec<OperationIr>),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub(crate) type ExecutionPlanId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Debug)]
pub(crate) struct ExecutionPlan<O> {
    /// The operations on which the exploration is related to.
    pub(crate) operations: Vec<OperationIr>,
    /// The criteria that signal when this plan should be executed. Only one trigger is necessary.
    pub(crate) triggers: Vec<ExecutionTrigger>,
    /// The optimization that should be used when executing this plan.
    pub(crate) optimization: BlockOptimization<O>,
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
mod required_operations_tests {
    use super::*;

    /// `()` stands in for an optimization: `required_operations` reads only the
    /// orderings, so the payload type is irrelevant to what is under test.
    fn ops(positions: &[usize]) -> ExecutionStrategy<()> {
        ExecutionStrategy::Operations { ordering: Arc::new(positions.to_vec()) }
    }

    /// THE CRASH, AS A UNIT TEST. `[1, 2, 3]` is the exact ordering the device
    /// runner died on. Three entries, highest index 3 — so it needs FOUR
    /// operations, and every guard that compared a COUNT read 3 and admitted a
    /// queue of 3.
    #[test]
    fn an_ordering_of_stream_positions_needs_more_than_its_own_length() {
        let positions = [1usize, 2, 3];
        let s = ops(&positions);
        assert_eq!(s.required_operations(), 4, "[1,2,3] reaches index 3");
        // THE DEFECT, STATED AS A RELATION RATHER THAN ASSERTED AS A CONSTANT:
        // the count both previous guards compared is strictly smaller than the
        // reach, so a queue sized by the count is admitted and then indexed
        // past its end. If these two ever coincide for this input the guard is
        // no longer distinguishable from the one it replaced.
        assert!(
            positions.len() < s.required_operations(),
            "a count of {} would admit a queue that cannot hold index {}",
            positions.len(),
            s.required_operations() - 1
        );
    }

    /// Contiguous-from-zero is the case where count and reach agree, which is
    /// why the count-based guard looked right for as long as it did.
    #[test]
    fn a_zero_based_ordering_is_the_case_where_a_count_happens_to_work() {
        assert_eq!(ops(&[0, 1, 2]).required_operations(), 3);
    }

    /// Order within the ordering must not change the answer — the reach is the
    /// maximum, not the last element.
    #[test]
    fn the_reach_is_the_maximum_not_the_final_entry() {
        assert_eq!(ops(&[3, 0, 2, 1]).required_operations(), 4);
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
        assert_eq!(c.required_operations(), 6, "member [5] reaches index 5");
    }

    /// An empty plan asks for nothing rather than underflowing on `max() + 1`.
    #[test]
    fn an_empty_ordering_requires_no_operations() {
        assert_eq!(ops(&[]).required_operations(), 0);
        assert_eq!(ExecutionStrategy::<()>::Composed(vec![]).required_operations(), 0);
    }
}
