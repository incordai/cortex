use cortex_core as cortex;

use cortex::module::Module;
use cortex::tensor::Tensor;

/// Applies the SiLU (Sigmoid Linear Unit) activation function element-wise.
///
/// See also [silu](cortex::tensor::activation::silu).
#[derive(Module, Debug, Default)]
pub struct SiLU;

impl SiLU {
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
        cortex::tensor::activation::silu(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = SiLU::new();

        assert_eq!(alloc::format!("{layer}"), "SiLU");
    }
}
