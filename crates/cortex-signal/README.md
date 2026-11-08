# Cortex Signal

> Signal processing operations for [Cortex](https://github.com/tracel-ai/burn) tensors

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-signal.svg)](https://crates.io/crates/cortex-signal)
[![Documentation](https://docs.rs/cortex-signal/badge.svg)](https://docs.rs/cortex-signal)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- FFTs: `rfft` and `irfft` are backend extensions with native kernels for Flex and the CubeCL
  backends, and are differentiable with `autodiff`. `cfft`, the complex FFT, is built
  from `rfft`.
- `stft` and `istft`.
- Windows: `hann_window`, `hamming_window` and `blackman_window`.

## Usage

Enable both `signal` and a backend feature on `cortex`, and use the operations through
`cortex::signal`:

```toml
cortex = { version = "0.22", features = ["signal", "flex"] }
```

No execution backend is enabled by default when depending on this crate directly; select `flex`,
`wgpu` or another backend feature on it. Signal processing previously lived at
`cortex_tensor::signal`; `cortex::tensor::signal` remains as a compatibility path.

## Remote Execution and Capture

FFTs are recorded as the custom operations `signal::rfft` and `signal::irfft`. Remote servers, and
interpreters replaying captured graphs, must enable the `router` feature and install
`register_fft_ops` in their custom-operation registry, then pass the registry to the server's
`with_custom_ops` or to `TensorInterpreter::with_custom_ops`.

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
