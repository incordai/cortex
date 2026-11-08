use super::qtensor::*;
use super::*;
use cortex_tensor::TensorData;
use cortex_tensor::Tolerance;

#[test]
fn should_support_tan_ops() {
    let tensor = QTensor::<2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.tan();
    let expected = TensorData::from([[0.0, 1.5574, -2.1850], [-0.1425, 1.1578, -3.3805]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
