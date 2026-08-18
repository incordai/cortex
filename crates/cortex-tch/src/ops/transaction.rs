use cortex_backend::distributed::DistributedOps;
use cortex_backend::ops::TransactionOps;

use crate::LibTorch;

impl TransactionOps<Self> for LibTorch {}

// DistributedOps has default implementations; LibTorch does not support collective operations.
impl DistributedOps<Self> for LibTorch {}
