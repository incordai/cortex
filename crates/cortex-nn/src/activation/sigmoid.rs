use cortex_core as cortex;

use cortex::module::Module;
use cortex::tensor::Tensor;

/// Applies the sigmoid function element-wise
/// See also [sigmoid](cortex::tensor::activation::sigmoid)
#[derive(Module, Debug, Default)]
pub struct Sigmoid;

impl Sigmoid {
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
        cortex::tensor::activation::sigmoid(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Sigmoid::new();

        assert_eq!(alloc::format!("{layer}"), "Sigmoid");
    }
}
