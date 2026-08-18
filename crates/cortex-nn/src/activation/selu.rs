use cortex_core as cortex;

use cortex::module::Module;
use cortex::tensor::Tensor;

/// Applies the Scaled Exponential Linear Unit function element-wise.
/// See also [selu](cortex::tensor::activation::selu)
#[derive(Module, Debug, Default)]
pub struct Selu;

impl Selu {
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
        cortex::tensor::activation::selu(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Selu::new();

        assert_eq!(alloc::format!("{layer}"), "Selu");
    }
}
