#![cfg(feature = "linalg")]

use cortex::tensor::Tensor;

#[test]
fn linalg_and_initializer_compatibility_paths_compile() {
    let _: cortex_linalg::Norm = cortex_linalg::Norm::L2;
    let _: cortex::linalg::Norm = cortex::linalg::Norm::L2;
    let _: cortex::tensor::linalg::Norm = cortex::tensor::linalg::Norm::L2;
    let _: cortex::module::Initializer = cortex::module::Initializer::Zeros;
    let _: cortex::nn::Initializer = cortex::nn::Initializer::Zeros;
    let _: cortex_nn::Initializer = cortex_nn::Initializer::Zeros;

    let _qr: fn(Tensor<2>, bool) -> (Tensor<2>, Tensor<2>) = cortex::linalg::qr::<2>;
}
