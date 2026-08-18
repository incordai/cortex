use cortex_tensor::Element;
use ctor::ctor;

// Re-export
use super::{FloatElem, IntElem};

#[ctor]
fn init_device_settings() {
    let mut device = cortex_tensor::Device::default();
    device
        .configure(
            cortex_tensor::DeviceConfig::default()
                .float_dtype(<FloatElem as Element>::dtype())
                .int_dtype(<IntElem as Element>::dtype()),
        )
        .unwrap();
}

/// Collection of types used across tests
#[allow(unused)]
pub mod prelude {
    pub use cortex_tensor::Tensor;

    use super::*;
    pub type TestTensor<const D: usize> = Tensor<D>;
    pub type TestTensorInt<const D: usize> = Tensor<D, cortex_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<D, cortex_tensor::Bool>;
}

#[allow(unused)]
pub use prelude::*;
