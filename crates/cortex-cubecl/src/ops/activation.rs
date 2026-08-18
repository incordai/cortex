use crate::{CubeBackend, CubeRuntime};
use cortex_backend::ops::ActivationOps;

impl<R: CubeRuntime> ActivationOps<Self> for CubeBackend<R> {}
