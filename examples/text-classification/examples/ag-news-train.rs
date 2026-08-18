#![recursion_limit = "256"]

// Only the CUDA `launch_multi` path uses these at the top level; the remote module imports
// them locally. Gating on both features avoids an unused-import warning for `remote,ddp`.
#[cfg(all(feature = "ddp", feature = "cuda"))]
use cortex::tensor::distributed::{DistributedConfig, ReduceOperation};
use cortex::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::{Device, DeviceConfig, Element},
    train::ExecutionStrategy,
};

use text_classification::{AgNewsDataset, training::ExperimentConfig};

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = cortex::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = cortex::tensor::flex32;

#[cfg(all(feature = "cuda", not(feature = "ddp")))]
pub fn launch_multi() {
    let mut devices = Device::enumerate(cortex::tensor::DeviceType::Cuda);
    devices
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    launch(ExecutionStrategy::MultiDevice(
        devices.into_vec(),
        cortex::train::MultiDeviceOptim::OptimSharded,
    ))
}

#[cfg(all(feature = "cuda", feature = "ddp"))]
pub fn launch_multi() {
    let mut devices = Device::enumerate(cortex::tensor::DeviceType::Cuda);
    devices
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    launch(ExecutionStrategy::ddp(
        devices.into_vec(),
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    ))
}

pub fn launch_single(mut device: Device) {
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    launch(ExecutionStrategy::SingleDevice(device))
}

pub fn launch(strategy: ExecutionStrategy) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(128, 512, 4, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<AgNewsDataset>(
        strategy,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use cortex::tensor::{Device, DeviceIndex};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = Device::libtorch_cuda(DeviceIndex::Default);
        #[cfg(target_os = "macos")]
        let device = Device::libtorch_mps();

        crate::launch_single(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use cortex::tensor::Device;

    pub fn run() {
        crate::launch_single(Device::libtorch());
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
mod wgpu {
    use cortex::tensor::{Device, DeviceKind};

    pub fn run() {
        crate::launch_single(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "remote")]
mod remote {
    #[cfg(feature = "ddp")]
    use crate::ElemType;
    #[cfg(feature = "ddp")]
    use cortex::tensor::distributed::{DistributedConfig, ReduceOperation};
    use cortex::tensor::{Device, DeviceType};
    #[cfg(feature = "ddp")]
    use cortex::tensor::{DeviceConfig, Element};
    #[cfg(feature = "ddp")]
    use cortex::train::ExecutionStrategy;

    /// Address of the `cortex-remote` server to train against.
    const ADDRESS: &str = "ws://localhost:3000";

    /// Train on a single one of the devices the remote server hosts.
    ///
    /// `launch_single` configures the device it receives, so don't configure the enumerated
    /// set here too — doing both locks the device's settings twice and returns
    /// [`DeviceError::AlreadyInitialized`](cortex::tensor::DeviceError::AlreadyInitialized).
    #[cfg(not(feature = "ddp"))]
    pub fn run() {
        let devices = Device::enumerate(DeviceType::remote_websocket(ADDRESS));
        crate::launch_single(devices.into_vec().pop().unwrap());
    }

    /// Same enumeration, but drive the devices with distributed data-parallel training.
    #[cfg(feature = "ddp")]
    pub fn run() {
        let mut devices = Device::enumerate(DeviceType::remote_websocket(ADDRESS));
        devices
            .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
            .unwrap();

        crate::launch(ExecutionStrategy::ddp(
            devices.into_vec(),
            DistributedConfig {
                all_reduce_op: ReduceOperation::Mean,
            },
        ));
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    pub fn run() {
        crate::launch_multi();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use cortex::tensor::{Device, DeviceIndex};

    pub fn run() {
        crate::launch_single(Device::rocm(DeviceIndex::Default));
    }
}

#[cfg(feature = "flex")]
mod flex {
    use cortex::tensor::Device;

    pub fn run() {
        crate::launch_single(Device::flex());
    }
}

fn main() {
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
    #[cfg(feature = "flex")]
    flex::run();
}
