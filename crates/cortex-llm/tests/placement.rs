//! Pipeline placement: layers spread over devices give the same result as
//! one device, with every layer computed where its weights live. Built with
//! both `flex` and a GPU feature, the two stages are the CPU and the GPU.

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::LayerPlacement;

fn devices() -> (Device, Device) {
    #[cfg(all(feature = "flex", feature = "cubecl-backend"))]
    {
        (Device::flex(), Device::default())
    }
    #[cfg(not(all(feature = "flex", feature = "cubecl-backend")))]
    {
        (Device::default(), Device::default())
    }
}

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

#[test]
fn layers_map_to_stages_in_order() {
    let (a, b) = devices();
    let p = LayerPlacement::even(vec![a.clone(), b.clone()], 5);
    assert_eq!(p.layers(), 5);
    let on_a: Vec<bool> = (0..5).map(|l| p.device_of(l) == &a).collect();
    if a != b {
        assert_eq!(on_a, [true, true, true, false, false], "3 then 2 layers");
    }
    assert_eq!(p.last_device(), &b);

    let p = LayerPlacement::by_counts(vec![a.clone(), b.clone()], &[1, 4]);
    assert_eq!(p.device_of(0), &a);
    assert!((1..5).all(|l| p.device_of(l) == &b));
}

#[test]
fn a_split_model_matches_one_device_and_runs_each_layer_in_place() {
    let (a, b) = devices();
    let (layers, d) = (4usize, 16usize);
    let weights: Vec<Vec<f32>> = (0..layers).map(|l| noise(d * d, 10 + l as u64)).collect();
    let input = noise(3 * d, 1);

    let run = |placement: &LayerPlacement| {
        let mut x = Tensor::<2>::from_data(TensorData::new(input.clone(), [3, d]), placement.device_of(0));
        for (l, w) in weights.iter().enumerate() {
            let w = Tensor::<2>::from_data(TensorData::new(w.clone(), [d, d]), placement.device_of(l));
            x = placement.hop(x, l);
            assert_eq!(&x.device(), placement.device_of(l), "layer {l} input is on its device");
            x = x.matmul(w).tanh();
        }
        x
    };

    let split = run(&LayerPlacement::by_counts(vec![a.clone(), b.clone()], &[2, 2]));
    assert_eq!(&split.device(), &b, "the output ends on the last stage");
    let whole = run(&LayerPlacement::single(a, layers));
    let (got, want) = (read(split), read(whole));
    let err = got.iter().zip(&want).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(err < 1e-5, "split and single-device results differ by {err}");
}
