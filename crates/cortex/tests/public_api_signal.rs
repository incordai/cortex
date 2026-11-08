#![cfg(feature = "signal")]

use cortex::tensor::Tensor;

#[test]
fn signal_compatibility_paths_compile() {
    let _: cortex_signal::StftOptions = cortex::signal::StftOptions::default();
    let _: cortex::tensor::signal::StftOptions = cortex::signal::StftOptions::default();
    let _rfft: fn(Tensor<1>, usize, Option<usize>) -> (Tensor<1>, Tensor<1>) = cortex::signal::rfft;
    let _irfft: fn(Tensor<1>, Tensor<1>, usize, Option<usize>) -> Tensor<1> =
        cortex::tensor::signal::irfft;
}
