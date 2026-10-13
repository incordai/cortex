//! The device matmuls against the host reference, on whichever backend this
//! test binary is built for (`--features flex`, `--features wgpu`, ...).
//!
//! The reference is `quant::dequantize` followed by a plain f32 matmul on the
//! host, so a decoding mistake in either the generic default (`ops.rs`) or a
//! kernel (`kernels/qmm.rs`) shows up as a numeric mismatch here.

use cortex_core::backend::Dispatch;
use cortex_core::tensor::{DType, Device, Int, Tensor, TensorData};
use cortex_llm::quant::{QuantFormat, QuantMode, dequantize, quantize};
use cortex_llm::{LlmOps, QuantizedExperts, QuantizedMatrix};

fn device() -> Device {
    Device::default()
}

fn noise(len: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * amp
        })
        .collect()
}

fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        for r in 0..k {
            let xv = x[i * k + r];
            for j in 0..n {
                y[i * n + j] += xv * w[r * n + j];
            }
        }
    }
    y
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 2e-4 * scale,
        "{label}: max |device - host| = {worst} against output scale {scale}"
    );
}

#[test]
fn quantized_matmul_matches_the_host_reference() {
    let dev = device();
    let (k, n) = (96, 128);
    let w = noise(k * n, 3, 0.05);
    for bits in [2u32, 3, 4, 5, 6, 8] {
        for mode in [QuantMode::Symmetric, QuantMode::Affine] {
            for group in [32usize, 64] {
                let f = QuantFormat::new(bits, group, mode).unwrap();
                let packed = quantize(&w, k, n, f).unwrap();
                let reference_w = dequantize(&packed);
                let mat = QuantizedMatrix::from_packed(&packed, &dev);
                // m = 1 is the decode path, 3 a small batch, 17 a prefill.
                for m in [1usize, 3, 17] {
                    let x = noise(m * k, 9 + m as u64, 1.0);
                    let want = host_matmul(&x, &reference_w, m, k, n);
                    let xt = Tensor::<2>::from_data(TensorData::new(x, [m, k]), &dev);
                    let got = mat.forward(xt).into_data().try_to_vec::<f32>().unwrap();
                    assert_close(&format!("bits {bits} {mode:?} g{group} m{m}"), &got, &want);
                }
            }
        }
    }
}

#[test]
fn gathered_matmul_uses_each_rows_own_expert() {
    let dev = device();
    let (experts, k, n) = (4usize, 64usize, 96usize);
    for (bits, mode) in [(4u32, QuantMode::Symmetric), (3, QuantMode::Affine), (8, QuantMode::Affine)] {
        let f = QuantFormat::new(bits, 32, mode).unwrap();
        let packed: Vec<_> = (0..experts)
            .map(|e| quantize(&noise(k * n, 100 + e as u64, 0.05), k, n, f).unwrap())
            .collect();
        let stacked = QuantizedExperts::from_packed(&packed, &dev).unwrap();

        // Repeats and gaps on purpose: expert 1 is never chosen.
        let choice = [2i64, 0, 3, 3, 0, 2];
        let t = choice.len();
        let x = noise(t * k, 77, 1.0);
        let mut want = Vec::with_capacity(t * n);
        for (row, &e) in choice.iter().enumerate() {
            let w = dequantize(&packed[e as usize]);
            want.extend(host_matmul(&x[row * k..(row + 1) * k], &w, 1, k, n));
        }

        let xt = Tensor::<2>::from_data(TensorData::new(x, [t, k]), &dev);
        let idx = Tensor::<1, Int>::from_data(TensorData::new(choice.to_vec(), [t]), &dev);
        let got = stacked.forward(xt, idx).into_data().try_to_vec::<f32>().unwrap();
        assert_close(&format!("gather bits {bits} {mode:?}"), &got, &want);
    }
}

/// The batched form the quantized KV cache reads through: one matrix per
/// batch entry, only the first `rows` of each buffer's `capacity` rows in
/// use, both as `x · W` and as `x · Wᵀ`, for decode-sized inputs (the
/// in-place kernels) and a prefill-sized one (decode, then dense matmul).
#[test]
fn batched_matmul_reads_the_used_rows_of_each_matrix_both_ways() {
    let dev = device();
    let (batch, capacity, rows, cols) = (3usize, 40usize, 37usize, 64usize);
    for bits in [2u32, 4, 8] {
        let format = QuantFormat::new(bits, 32, QuantMode::Affine).unwrap();
        let packed: Vec<_> = (0..batch)
            .map(|b| quantize(&noise(capacity * cols, 40 + b as u64, 1.0), capacity, cols, format).unwrap())
            .collect();
        let dense: Vec<Vec<f32>> = packed.iter().map(dequantize).collect();
        let (wpr, gpr) = (format.words_per_row(cols), format.groups_per_row(cols));
        let data: Vec<u32> = packed.iter().flat_map(|p| p.data.clone()).collect();
        let scales: Vec<f32> = packed.iter().flat_map(|p| p.scales.clone()).collect();
        let biases: Vec<f32> = packed.iter().flat_map(|p| p.biases.clone().unwrap()).collect();
        let data = Tensor::<3, Int>::from_data(TensorData::new(data, [batch, capacity, wpr]), (&dev, DType::U32));
        let scales = Tensor::<3>::from_data(TensorData::new(scales, [batch, capacity, gpr]), &dev);
        let biases = Tensor::<3>::from_data(TensorData::new(biases, [batch, capacity, gpr]), &dev);

        for m in [1usize, 3, 8, 11] {
            for transpose in [false, true] {
                let k = if transpose { cols } else { rows };
                let x = noise(batch * m * k, 7 + m as u64, 1.0);
                let mut want = Vec::new();
                for (b, w) in dense.iter().enumerate() {
                    let xb = &x[b * m * k..(b + 1) * m * k];
                    if transpose {
                        // x[m, cols] · W[..rows]ᵀ
                        let mut wt = vec![0f32; cols * rows];
                        for j in 0..rows {
                            for c in 0..cols {
                                wt[c * rows + j] = w[j * cols + c];
                            }
                        }
                        want.extend(host_matmul(xb, &wt, m, cols, rows));
                    } else {
                        want.extend(host_matmul(xb, &w[..rows * cols], m, rows, cols));
                    }
                }
                let xt = Tensor::<3>::from_data(TensorData::new(x, [batch, m, k]), &dev);
                let y: Tensor<3> = Tensor::from_dispatch(<Dispatch as LlmOps>::batched_quantized_matmul(
                    xt.into_dispatch(),
                    data.clone().into_dispatch(),
                    scales.clone().into_dispatch(),
                    biases.clone().into_dispatch(),
                    rows,
                    bits,
                    32,
                    true,
                    transpose,
                ));
                let n = if transpose { rows } else { cols };
                assert_eq!(y.dims(), [batch, m, n]);
                let got = y.into_data().try_to_vec::<f32>().unwrap();
                assert_close(&format!("batched bits {bits} m {m} transpose {transpose}"), &got, &want);
            }
        }
    }
}
