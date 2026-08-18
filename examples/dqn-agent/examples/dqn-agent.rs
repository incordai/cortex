use cortex::tensor::Device;

#[allow(unreachable_code)]
fn select_device() -> Device {
    #[cfg(not(any(
        feature = "tch-gpu",
        feature = "tch-cpu",
        feature = "wgpu",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm",
        feature = "cuda",
    )))]
    return Device::flex();

    #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    return Device::libtorch_cuda(cortex::tensor::DeviceIndex::Default);

    #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    return Device::libtorch_mps();

    #[cfg(feature = "tch-cpu")]
    return Device::libtorch();

    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    return Device::wgpu(cortex::tensor::DeviceKind::DefaultDevice);

    #[cfg(feature = "cuda")]
    return Device::cuda(cortex::tensor::DeviceIndex::Default);

    #[cfg(feature = "rocm")]
    return Device::rocm(cortex::tensor::DeviceIndex::Default);

    unreachable!("At least one backend will be selected.")
}

fn main() {
    let device = select_device();
    dqn_agent::training::run(device.autodiff());
}
