//! Packed quantized weights on a device, and the matmuls that read them.
//!
//! Entry points for model code. Both types validate shapes once, at
//! construction, and forward through [`LlmOps`](crate::LlmOps) on
//! `Dispatch`, so the backend picks the kernel: CubeCL runtimes run the
//! packed kernels in `kernels/qmm.rs`, every other backend decodes with the
//! generic default in `ops.rs`.

use alloc::vec::Vec;
use cortex_core::backend::Dispatch;
use cortex_core::tensor::{DType, Device, Int, Tensor, TensorData};

use crate::LlmOps;
use crate::quant::{PackedWeight, QuantError, QuantFormat, QuantMode};

/// A packed `[k, n]` weight resident on a device.
#[derive(Clone, Debug)]
pub struct QuantizedMatrix {
    data: Tensor<2, Int>,
    scales: Tensor<2>,
    biases: Option<Tensor<2>>,
    format: QuantFormat,
    k: usize,
    n: usize,
}

impl QuantizedMatrix {
    /// Uploads a host-side packed weight.
    pub fn from_packed(packed: &PackedWeight, device: &Device) -> Self {
        let (k, n, f) = (packed.k, packed.n, packed.format);
        let wpr = f.words_per_row(n);
        let gpr = f.groups_per_row(n);
        // Explicitly `u32`: the default int dtype is signed and cannot hold
        // a word with its top bit set.
        let data = Tensor::<2, Int>::from_data(
            TensorData::new(packed.data.clone(), [k, wpr]),
            (device, DType::U32),
        );
        let scales = Tensor::<2>::from_data(TensorData::new(packed.scales.clone(), [k, gpr]), device);
        let biases = packed
            .biases
            .as_ref()
            .map(|b| Tensor::<2>::from_data(TensorData::new(b.clone(), [k, gpr]), device));
        Self {
            data,
            scales,
            biases,
            format: f,
            k,
            n,
        }
    }

    /// Wraps tensors already on the device, checking every shape against the
    /// format. `data` must hold `u32` words.
    pub fn from_tensors(
        data: Tensor<2, Int>,
        scales: Tensor<2>,
        biases: Option<Tensor<2>>,
        format: QuantFormat,
    ) -> Result<Self, QuantError> {
        if data.dtype() != DType::U32 {
            return Err(QuantError::DataNotU32);
        }
        let [k, wpr] = data.dims();
        let n = wpr * 32 / format.bits() as usize;
        format.check_columns(n)?;
        if wpr != format.words_per_row(n) {
            return Err(QuantError::LengthMismatch {
                what: "data words per row",
                expected: format.words_per_row(n),
                actual: wpr,
            });
        }
        let expected = [k, format.groups_per_row(n)];
        check_dims("scales", expected, scales.dims())?;
        match (format.mode(), &biases) {
            (QuantMode::Affine, Some(b)) => check_dims("biases", expected, b.dims())?,
            (QuantMode::Affine, None) => {
                return Err(QuantError::LengthMismatch {
                    what: "biases",
                    expected: k * format.groups_per_row(n),
                    actual: 0,
                });
            }
            (QuantMode::Symmetric, Some(b)) => {
                return Err(QuantError::LengthMismatch {
                    what: "biases",
                    expected: 0,
                    actual: b.dims().iter().product(),
                });
            }
            (QuantMode::Symmetric, None) => {}
        }
        Ok(Self {
            data,
            scales,
            biases,
            format,
            k,
            n,
        })
    }

    /// Input features.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Output features.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The packing format.
    pub fn format(&self) -> QuantFormat {
        self.format
    }

    /// The weight decoded to a dense `[k, n]` tensor (in the dtype of the
    /// scales). Callers: `qlora.rs` (`QLoraLinear::fuse`), and anything that
    /// needs the full-precision weight back, e.g. export.
    pub fn dequantize(&self) -> Tensor<2> {
        let affine = self.format.mode() == QuantMode::Affine;
        let biases = self.biases.clone().unwrap_or_else(|| self.scales.clone());
        Tensor::from_dispatch(<Dispatch as LlmOps>::dequantize_packed(
            self.data.clone().into_dispatch(),
            self.scales.clone().into_dispatch(),
            biases.into_dispatch(),
            self.format.bits(),
            self.format.group_size(),
            affine,
        ))
    }

    /// `x[m, k] · W` → `[m, n]`.
    ///
    /// # Panics
    /// If `x` does not have `k` columns.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let [_, xk] = x.dims();
        assert_eq!(xk, self.k, "input has {xk} features, the weight expects {}", self.k);
        let affine = self.format.mode() == QuantMode::Affine;
        let biases = self.biases.clone().unwrap_or_else(|| self.scales.clone());
        Tensor::from_dispatch(<Dispatch as LlmOps>::quantized_matmul(
            x.into_dispatch(),
            self.data.clone().into_dispatch(),
            self.scales.clone().into_dispatch(),
            biases.into_dispatch(),
            self.format.bits(),
            self.format.group_size(),
            affine,
        ))
    }
}

/// `e` packed `[k, n]` experts stacked into one `[e, k, n]` weight.
#[derive(Clone, Debug)]
pub struct QuantizedExperts {
    data: Tensor<3, Int>,
    scales: Tensor<3>,
    biases: Option<Tensor<3>>,
    format: QuantFormat,
    experts: usize,
    k: usize,
    n: usize,
}

impl QuantizedExperts {
    /// Uploads `experts`, which must all share one shape and format.
    pub fn from_packed(experts: &[PackedWeight], device: &Device) -> Result<Self, QuantError> {
        let first = experts.first().ok_or(QuantError::LengthMismatch {
            what: "experts",
            expected: 1,
            actual: 0,
        })?;
        let (k, n, f) = (first.k, first.n, first.format);
        let (wpr, gpr) = (f.words_per_row(n), f.groups_per_row(n));
        let mut data = Vec::with_capacity(experts.len() * k * wpr);
        let mut scales = Vec::with_capacity(experts.len() * k * gpr);
        let mut biases = Vec::new();
        for e in experts {
            if (e.k, e.n, e.format) != (k, n, f) {
                return Err(QuantError::LengthMismatch {
                    what: "expert shape",
                    expected: k * n,
                    actual: e.k * e.n,
                });
            }
            data.extend_from_slice(&e.data);
            scales.extend_from_slice(&e.scales);
            if let Some(b) = &e.biases {
                biases.extend_from_slice(b);
            }
        }
        let count = experts.len();
        Ok(Self {
            data: Tensor::<3, Int>::from_data(
                TensorData::new(data, [count, k, wpr]),
                (device, DType::U32),
            ),
            scales: Tensor::<3>::from_data(TensorData::new(scales, [count, k, gpr]), device),
            biases: (f.mode() == QuantMode::Affine)
                .then(|| Tensor::<3>::from_data(TensorData::new(biases, [count, k, gpr]), device)),
            format: f,
            experts: count,
            k,
            n,
        })
    }

    /// Number of experts.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Row `t` of `x[t, k]` times expert `indices[t]` → `[t, n]`.
    ///
    /// # Panics
    /// If `x` does not have `k` columns or `indices` does not have one entry
    /// per row of `x`.
    pub fn forward(&self, x: Tensor<2>, indices: Tensor<1, Int>) -> Tensor<2> {
        let [t, xk] = x.dims();
        assert_eq!(xk, self.k, "input has {xk} features, the experts expect {}", self.k);
        assert_eq!(indices.dims()[0], t, "one expert index is needed per input row");
        let affine = self.format.mode() == QuantMode::Affine;
        let biases = self.biases.clone().unwrap_or_else(|| self.scales.clone());
        Tensor::from_dispatch(<Dispatch as LlmOps>::gather_quantized_matmul(
            x.into_dispatch(),
            self.data.clone().into_dispatch(),
            self.scales.clone().into_dispatch(),
            biases.into_dispatch(),
            indices.into_dispatch(),
            self.format.bits(),
            self.format.group_size(),
            affine,
        ))
    }

    /// Output features of every expert.
    pub fn n(&self) -> usize {
        self.n
    }
}

fn check_dims<const D: usize>(
    what: &'static str,
    expected: [usize; D],
    actual: [usize; D],
) -> Result<(), QuantError> {
    if expected == actual {
        Ok(())
    } else {
        Err(QuantError::LengthMismatch {
            what,
            expected: expected.iter().product(),
            actual: actual.iter().product(),
        })
    }
}
