//! Gradients through the LLM ops and QLoRA training, on an autodiff device
//! over whichever backend this test binary is built for.
//!
//! Every gradient is checked against an oracle that shares no code with the
//! backward passes in `src/autodiff.rs`: host dequantization for the matmuls,
//! the host RoPE reference for the rotation, and built-in autodiff through
//! primitive tensor ops for RMS norm.
#![cfg(feature = "autodiff")]

use cortex_core::tensor::{Device, Int, Tensor, TensorData};
use cortex_llm::quant::{self, QuantFormat, QuantMode};
use cortex_llm::{QLoraLinear, QuantizedExperts, QuantizedMatrix, rms_norm, rope, rope_reference};

fn dev() -> Device {
    Device::default().autodiff()
}

fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    match t.try_into_data() {
        Ok(d) => d.convert::<f32>().try_to_vec::<f32>().unwrap(),
        Err(e) => panic!("device execution failed: {e:#?}"),
    }
}

fn tensor<const D: usize>(v: Vec<f32>, shape: [usize; D]) -> Tensor<D> {
    Tensor::from_data(TensorData::new(v, shape), &dev())
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(err <= tol * scale, "{what}: max error {err} (scale {scale})");
}

/// `dL/dx = c · Wᵀ` for `L = Σ c ⊙ (x · W)`, with `W` decoded on the host.
fn host_input_grad(c: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut g = vec![0f32; m * k];
    for r in 0..m {
        for i in 0..k {
            g[r * k + i] = (0..n).map(|j| c[r * n + j] * w[i * n + j]).sum();
        }
    }
    g
}

#[test]
fn quantized_matmul_propagates_the_input_gradient() {
    let (k, n) = (64usize, 96usize);
    for mode in [QuantMode::Affine, QuantMode::Symmetric] {
        let format = QuantFormat::new(4, 16, mode).unwrap();
        let packed = quant::quantize(&noise(k * n, 3), k, n, format).unwrap();
        let w = quant::dequantize(&packed);
        let mat = QuantizedMatrix::from_packed(&packed, &dev());
        // Decode rows (packed kernel) and prefill rows (dequantize + matmul).
        for m in [3usize, 12] {
            let x = tensor(noise(m * k, 7 + m as u64), [m, k]).require_grad();
            let c = noise(m * n, 11);
            let loss = (mat.forward(x.clone()) * tensor(c.clone(), [m, n])).sum();
            let grads = loss.backward();
            let got = read(x.grad(&grads).expect("x is tracked"));
            assert_close(&got, &host_input_grad(&c, &w, m, k, n), 1e-4, &format!("{mode:?} m={m}"));
        }
    }
}

#[test]
fn gathered_quantized_matmul_routes_each_row_to_its_expert() {
    let (e, k, n, t) = (3usize, 32usize, 32usize, 5usize);
    let format = QuantFormat::new(8, 16, QuantMode::Affine).unwrap();
    let packed: Vec<_> = (0..e)
        .map(|i| quant::quantize(&noise(k * n, 20 + i as u64), k, n, format).unwrap())
        .collect();
    let experts = QuantizedExperts::from_packed(&packed, &dev()).unwrap();
    let idx = [2i64, 0, 2, 1, 0];
    let indices = Tensor::<1, Int>::from_data(TensorData::new(idx.to_vec(), [t]), &dev());

    let x = tensor(noise(t * k, 5), [t, k]).require_grad();
    let c = noise(t * n, 6);
    let loss = (experts.forward(x.clone(), indices) * tensor(c.clone(), [t, n])).sum();
    let got = read(x.grad(&loss.backward()).expect("x is tracked"));

    let mut want = Vec::with_capacity(t * k);
    for (row, &ex) in idx.iter().enumerate() {
        let w = quant::dequantize(&packed[ex as usize]);
        want.extend(host_input_grad(&c[row * n..(row + 1) * n], &w, 1, k, n));
    }
    assert_close(&got, &want, 1e-4, "gathered input gradient");
}

#[test]
fn rms_norm_gradients_match_autodiff_through_primitive_ops() {
    let (rows, d, eps) = (6usize, 40usize, 1e-5f32);
    let xv = noise(rows * d, 1);
    let gv = noise(d, 2);
    let c = tensor(noise(rows * d, 3), [2, 3, d]);
    for unit_offset in [false, true] {
        // The fused op.
        let x = tensor(xv.clone(), [2, 3, d]).require_grad();
        let g = tensor(gv.clone(), [d]).require_grad();
        let loss = (rms_norm(x.clone(), g.clone(), eps, unit_offset) * c.clone()).sum();
        let grads = loss.backward();
        let (dx, dg) = (read(x.grad(&grads).unwrap()), read(g.grad(&grads).unwrap()));

        // The same function written with primitive ops, differentiated by
        // the built-in backward passes.
        let x2 = tensor(xv.clone(), [2, 3, d]).require_grad();
        let g2 = tensor(gv.clone(), [d]).require_grad();
        let w = if unit_offset { g2.clone() + 1.0 } else { g2.clone() };
        let inv = (x2.clone() * x2.clone()).mean_dim(2).add_scalar(eps).sqrt().recip();
        let y = x2.clone() * inv * w.reshape([1, 1, d]);
        let grads2 = (y * c.clone()).sum().backward();

        let tag = format!("unit_offset={unit_offset}");
        assert_close(&dx, &read(x2.grad(&grads2).unwrap()), 1e-4, &format!("dx {tag}"));
        assert_close(&dg, &read(g2.grad(&grads2).unwrap()), 1e-4, &format!("dgamma {tag}"));
    }
}

#[test]
fn rope_gradient_is_the_transposed_rotation() {
    let (s, d, offset) = (3usize, 8usize, 5usize);
    let inv: Vec<f32> = (0..3).map(|p| 10000f32.powf(-2.0 * p as f32 / 6.0)).collect();
    for traditional in [false, true] {
        let x = tensor(noise(s * d, 4), [1, 1, s, d]).require_grad();
        let c = noise(s * d, 9);
        let freqs = tensor(inv.clone(), [inv.len()]);
        let y = rope(x.clone(), freqs, offset, traditional);
        let got = read(x.grad(&(y * tensor(c.clone(), [1, 1, s, d])).sum().backward()).unwrap());

        // Column j of the rotation is the reference applied to basis vector
        // j, so dL/dx_j = Σ_i c_i R_ij.
        let want: Vec<f32> = (0..s * d)
            .map(|j| {
                let mut e = vec![0f32; s * d];
                e[j] = 1.0;
                let col = rope_reference(&e, s, d, &inv, offset, traditional);
                col.iter().zip(&c).map(|(r, ci)| r * ci).sum()
            })
            .collect();
        assert_close(&got, &want, 1e-4, &format!("traditional={traditional}"));
    }
}

#[test]
fn qlora_trains_only_the_adapter_and_fuses_back_to_a_packed_weight() {
    let (m, k, n, rank) = (4usize, 32usize, 32usize, 4usize);
    let format = QuantFormat::new(8, 32, QuantMode::Affine).unwrap();
    let packed = quant::quantize(&noise(k * n, 30), k, n, format).unwrap();
    let base = QuantizedMatrix::from_packed(&packed, &dev());
    let mut layer = QLoraLinear::new(base.clone(), rank, 8.0, &dev());
    let xv = noise(m * k, 31);
    let x = tensor(xv.clone(), [m, k]);

    // b = 0 at initialization: the layer is exactly the base.
    assert_close(&read(layer.forward(x.clone())), &read(base.forward(x.clone())), 1e-6, "init");

    // dL/db = scale · (x·a)ᵀ · c, checked on the host from the live factors.
    let c = noise(m * n, 32);
    let loss = (layer.forward(x.clone()) * tensor(c.clone(), [m, n])).sum();
    let grads = loss.backward();
    let a = read(layer.adapter.a.val());
    let xa: Vec<f32> = (0..m * rank)
        .map(|i| (0..k).map(|j| xv[(i / rank) * k + j] * a[j * rank + i % rank]).sum())
        .collect();
    let scale = layer.adapter.scale as f32;
    let want_db: Vec<f32> = (0..rank * n)
        .map(|i| scale * (0..m).map(|r| xa[r * rank + i / n] * c[r * n + i % n]).sum::<f32>())
        .collect();
    let db = read(layer.adapter.b.val().grad(&grads).expect("b trains"));
    assert_close(&db, &want_db, 1e-4, "dL/db");
    // b is zero, so nothing reaches a yet; it is still tracked.
    let da = read(layer.adapter.a.val().grad(&grads).expect("a trains"));
    assert!(da.iter().all(|v| *v == 0.0), "dL/da is zero while b is zero");

    // Move b off zero and check the fused, re-packed weight reproduces the
    // adapted layer to within one more rounding of the 8-bit grid.
    let b = noise(rank * n, 33).iter().map(|v| v * 0.05).collect::<Vec<_>>();
    layer.adapter.b = cortex_core::module::Param::from_tensor(tensor(b, [rank, n]));
    let adapted = read(layer.forward(x.clone()));
    let fused = QuantizedMatrix::from_packed(&layer.fuse().unwrap(), &dev());
    assert_close(&read(fused.forward(x.clone())), &adapted, 2e-2, "fused vs adapted");
    let unadapted = read(base.forward(x));
    assert!(
        adapted.iter().zip(&unadapted).any(|(a, b)| (a - b).abs() > 1e-3),
        "the adapter must change the output for the fused check to mean anything"
    );
}
