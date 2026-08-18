use cortex::tensor::Device;
use simple_regression::{inference, training};

static ARTIFACT_DIR: &str = "/tmp/cortex-example-regression";

#[cfg(feature = "flex")]
mod flex {
    use cortex::tensor::Device;

    pub fn run() {
        super::run(Device::flex());
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use cortex::tensor::{Device, DeviceIndex};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = Device::libtorch_cuda(DeviceIndex::Default);
        #[cfg(target_os = "macos")]
        let device = Device::libtorch_mps();

        super::run(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use cortex::tensor::{Device, DeviceKind};

    pub fn run() {
        super::run(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use cortex::tensor::Device;
    pub fn run() {
        super::run(Device::libtorch());
    }
}

// #[cfg(feature = "remote")]
// mod remote {
//     use cortex::backend::{RemoteBackend, remote::RemoteDevice};

//     pub fn run() {
//         let device = RemoteDevice::default();
//         super::run::<RemoteBackend>(device);
//     }
// }

/// Train a regression model and predict results on a number of samples.
pub fn run(device: Device) {
    training::run(ARTIFACT_DIR, device.clone());
    inference::infer(ARTIFACT_DIR, device)
}

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "remote")]
    remote::run();
}
