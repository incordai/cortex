use crate::NdArray;
use cortex_backend::distributed::DistributedOps;
use cortex_backend::ops::TransactionOps;

impl TransactionOps<Self> for NdArray {}

// DistributedOps has default implementations; NdArray does not support collective operations.
impl DistributedOps<Self> for NdArray {}
