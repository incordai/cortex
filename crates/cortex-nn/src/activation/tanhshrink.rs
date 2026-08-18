use cortex_core as cortex;

use cortex::module::Module;
use cortex::tensor::Tensor;

/// Applies the Tanhshrink function element-wise, `x - tanh(x)`.
/// See also [tanhshrink](cortex::tensor::activation::tanhshrink)
///
#[derive(Module, Debug, Default)]
pub struct Tanhshrink;

impl Tanhshrink {
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
        cortex::tensor::activation::tanhshrink(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Tanhshrink::new();

        assert_eq!(alloc::format!("{layer}"), "Tanhshrink");
    }
}
