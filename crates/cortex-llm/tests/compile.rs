//! `compile`: the compiled function agrees with the eager one for changing
//! inputs and shapes, hands out outputs a later call cannot change, and on
//! a hardware-graph backend (CUDA) really serves calls from graph replays.

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::compile;

fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.5
        })
        .collect()
}

fn read(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn compiled_calls_match_eager_ones_and_do_not_alias() {
    let device = Device::default();
    let (k, n) = (16usize, 8usize);
    let w = Tensor::<2>::from_data(TensorData::new(noise(k * n, 1), [k, n]), &device);
    let step = {
        let w = w.clone();
        move |xs: &[Tensor<2>]| (xs[0].clone().mul_scalar(2.0).add_scalar(1.0)).matmul(w.clone()).tanh() + xs[1].clone()
    };
    let eager = |x: &Tensor<2>, b: &Tensor<2>| read(step.clone()(&[x.clone(), b.clone()]));
    let mut compiled = compile(&device, step.clone());

    let mut kept = Vec::new();
    for (call, m) in [(0u64, 4usize), (1, 4), (2, 4), (3, 6), (4, 6), (5, 4)] {
        let x = Tensor::<2>::from_data(TensorData::new(noise(m * k, 10 + call), [m, k]), &device);
        let b = Tensor::<2>::from_data(TensorData::new(noise(m * n, 20 + call), [m, n]), &device);
        let want = eager(&x, &b);
        let got_tensor = compiled.call(&[x, b]);
        let got = read(got_tensor.clone());
        assert!(max_err(&got, &want) < 1e-5, "call {call} (m = {m}) differs from eager");
        kept.push((got_tensor, want));
    }
    assert_eq!(compiled.signatures(), 2, "one recording per input shape");

    // Every output handed out still holds its own call's result.
    for (i, (tensor, want)) in kept.into_iter().enumerate() {
        assert!(max_err(&read(tensor), &want) < 1e-5, "output {i} changed after later calls");
    }

    #[cfg(feature = "cuda")]
    assert!(
        compiled.hardware_replays() >= 3,
        "on CUDA, repeated shapes must be served by graph replays (got {})",
        compiled.hardware_replays()
    );
}
