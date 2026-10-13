//! RMS norm and RoPE on the device against the host references, on whichever
//! backend this test binary is built for (`--features flex`, `wgpu`, ...).

use cortex_core::tensor::{DType, Device, Tensor, TensorData};
use cortex_llm::{rms_norm, rms_norm_reference, rope, rope_inv_freq, rope_reference};

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

/// Reads a device tensor back, surfacing a failed execution with its full
/// error chain instead of the generic read panic.
fn read<const D: usize>(t: Tensor<D>, label: &str) -> Vec<f32> {
    match t.try_into_data() {
        Ok(data) => data.try_to_vec::<f32>().unwrap(),
        Err(e) => panic!("{label}: device execution failed: {e:#?}"),
    }
}

fn max_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()))
}

#[test]
fn rms_norm_matches_the_reference() {
    let dev = Device::default();
    // d = 2048 is a model width; 100 is not a multiple of the block size; 1
    // and 300 rows cover decode and a batch.
    for (rows, d) in [(1usize, 2048usize), (300, 2048), (7, 100)] {
        let x = noise(rows * d, 3);
        let gamma = noise(d, 4);
        for unit_offset in [false, true] {
            let want = rms_norm_reference(&x, &gamma, 1e-6, unit_offset);
            let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [rows, d]), &dev);
            let gt = Tensor::<1>::from_data(TensorData::new(gamma.clone(), [d]), &dev);
            let got = read(rms_norm(xt, gt, 1e-6, unit_offset), "rms_norm");
            let err = max_err(&got, &want);
            assert!(err < 1e-4, "rows {rows} d {d} offset {unit_offset}: max error {err}");
        }
    }
}

#[test]
fn rope_matches_the_reference() {
    let dev = Device::default();
    // [batch, heads, seq, head_dim]; partial rotation (rot < d) and full.
    let (b, h, s, d) = (2usize, 3usize, 5usize, 64usize);
    let x = noise(b * h * s * d, 9);
    for (rot, traditional) in [(64usize, false), (64, true), (32, false), (16, true)] {
        for offset in [0usize, 37] {
            let inv: Vec<f32> = (0..rot / 2)
                .map(|p| 10000f32.powf(-2.0 * p as f32 / rot as f32))
                .collect();
            let want = rope_reference(&x, s, d, &inv, offset, traditional);
            let xt = Tensor::<4>::from_data(TensorData::new(x.clone(), [b, h, s, d]), &dev);
            let freqs = rope_inv_freq(rot, 10000.0, &dev);
            let got = read(rope(xt, freqs, offset, traditional), "rope");
            let err = max_err(&got, &want);
            assert!(
                err < 1e-4,
                "rot {rot} traditional {traditional} offset {offset}: max error {err}"
            );
        }
    }
}

/// A rotated pair keeps its length; pass-through dims are untouched. Checks
/// the behaviour directly rather than against the reference, so a reference
/// that shared a bug with the op would still be caught.
#[test]
fn rope_rotates_pairs_and_leaves_the_rest() {
    let dev = Device::default();
    let (s, d, rot) = (4usize, 8usize, 4usize);
    let x = noise(s * d, 21);
    let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [s, d]), &dev);
    let freqs = Tensor::<1>::from_data(TensorData::new(vec![0.5f32, 0.25], [2]), (&dev, DType::F32));
    let got = rope(xt, freqs, 3, false).into_data().try_to_vec::<f32>().unwrap();
    for t in 0..s {
        let row = &x[t * d..(t + 1) * d];
        let out = &got[t * d..(t + 1) * d];
        for p in 0..rot / 2 {
            let before = row[p].hypot(row[p + rot / 2]);
            let after = out[p].hypot(out[p + rot / 2]);
            assert!((before - after).abs() < 1e-5, "pair {p} at row {t} changed length");
        }
        assert_eq!(&out[rot..], &row[rot..], "pass-through dims changed at row {t}");
    }
}
