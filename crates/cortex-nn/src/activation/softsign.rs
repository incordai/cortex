use cortex_core as cortex;

use cortex::module::Module;
use cortex::tensor::Tensor;

/// Applies the softsign function element-wise
/// See also [softsign](cortex::tensor::activation::softsign)
#[derive(Module, Debug, Default)]
pub struct Softsign;

impl Softsign {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        cortex::tensor::activation::softsign(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Softsign::new();

        assert_eq!(alloc::format!("{layer}"), "Softsign");
    }
}
