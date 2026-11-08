#[cfg(feature = "flex")]
mod flex {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, VisionBackend};
    use cortex_core::backend::Flex;

    impl BoolVisionOps for Flex {}
    impl IntVisionOps for Flex {}
    impl FloatVisionOps for Flex {}
    impl VisionBackend for Flex {}
}
