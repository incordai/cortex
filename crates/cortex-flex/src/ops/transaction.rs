//! Transaction operations for the Flex backend.

use crate::Flex;
use cortex_backend::distributed::DistributedOps;
use cortex_backend::ops::TransactionOps;

// TransactionOps has default implementations.
impl TransactionOps<Flex> for Flex {}

// DistributedOps has default implementations; Flex does not support collective operations.
impl DistributedOps<Flex> for Flex {}
