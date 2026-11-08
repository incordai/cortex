use super::qtensor::*;
use super::*;
use cortex_tensor::TensorData;
use cortex_tensor::Tolerance;

#[test]
fn should_support_abs_ops() {
    let tensor = QTensor::<2>::int8([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

    let output = tensor.abs();

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            Tolerance::absolute(1e-1),
        );
}
