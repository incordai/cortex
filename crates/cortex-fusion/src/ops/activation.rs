use crate::{Fusion, FusionBackend};
use cortex_backend::ops::ActivationOps;

impl<B: FusionBackend> ActivationOps<Self> for Fusion<B> {}
