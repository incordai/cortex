use cortex::tensor::Device;

pub fn launch(device: Device) {
    modern_lstm::inference::infer("/tmp/modern-lstm", device);
}

#[cfg(feature = "flex")]
mod flex {
    use cortex::tensor::Device;

    pub fn run() {
        crate::launch(Device::flex());
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use cortex::tensor::{Device, DeviceKind};

    pub fn run() {
        crate::launch(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use cortex::tensor::{Device, DeviceIndex};

    pub fn run() {
        crate::launch(Device::cuda(DeviceIndex::Default));
    }
}

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
}
