//! The quantized KV cache on whichever backend this test binary is built
//! for, checked against the host quantizer in `quant.rs` (an independent
//! implementation of the same format) and against the full-precision cache.

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::quant::{self, QuantFormat, QuantMode};
use cortex_llm::{KvCache, QuantizedKvCache, cache_attention};

fn dev() -> Device {
    Device::default()
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

fn kv(v: Vec<f32>, h: usize, n: usize, d: usize) -> Tensor<4> {
    Tensor::from_data(TensorData::new(v, [1, h, n, d]), &dev())
}

/// What the host quantizer makes of `[rows, d]` values.
fn host_round_trip(v: &[f32], rows: usize, d: usize, bits: u32, group: usize) -> Vec<f32> {
    let format = QuantFormat::new(bits, group, QuantMode::Affine).unwrap();
    quant::dequantize(&quant::quantize(v, rows, d, format).unwrap())
}

#[test]
fn every_bit_width_round_trips_like_the_host_quantizer_across_growth() {
    let (h, d, group) = (2usize, 64usize, 32usize);
    for bits in [2u32, 4, 8] {
        let mut cache = QuantizedKvCache::new(bits, group).unwrap().with_step(4);
        let mut written: Vec<Vec<f32>> = vec![Vec::new(); h];
        let mut seed = 0;
        for n in [3usize, 3, 1, 5] {
            seed += 1;
            let k = noise(h * n * d, seed);
            for (head, rows) in written.iter_mut().enumerate() {
                rows.extend_from_slice(&k[head * n * d..(head + 1) * n * d]);
            }
            let (keys, values) = cache.update(kv(k.clone(), h, n, d), kv(k, h, n, d));
            let len = cache.offset();
            let want: Vec<f32> = written
                .iter()
                .flat_map(|rows| host_round_trip(rows, len, d, bits, group))
                .collect();
            let got = read(keys);
            let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-5, "{bits}-bit keys after {len} positions: max error {err}");
            assert_eq!(read(values), got, "{bits}-bit values took a different path");
        }
    }
}

#[test]
fn trimmed_positions_are_overwritten_and_memory_shrinks() {
    let (h, d) = (1usize, 64usize);
    let mut cache = QuantizedKvCache::new(8, 64).unwrap().with_step(8);
    let first = noise(5 * d, 1);
    cache.update(kv(first.clone(), h, 5, d), kv(first.clone(), h, 5, d));
    assert_eq!(cache.trim(2), 2);
    let late = noise(d, 2);
    let (keys, _) = cache.update(kv(late.clone(), h, 1, d), kv(late.clone(), h, 1, d));
    assert_eq!(cache.offset(), 4);

    let mut rows = first[..3 * d].to_vec();
    rows.extend_from_slice(&late);
    let want = host_round_trip(&rows, 4, d, 8, 64);
    let got = read(keys);
    let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(err < 1e-5, "max error {err}");

    // 8 positions of capacity, f32 input: 1 byte per value plus two f32 per
    // 64 values, against 4 bytes per value for the full-precision cache.
    let dense = 2 * 8 * d * 4;
    assert_eq!(cache.nbytes(), 2 * 8 * (d + 2 * 4));
    assert!(cache.nbytes() * 3 < dense, "{} bytes vs {dense}", cache.nbytes());
}

#[test]
fn attention_over_the_quantized_cache_tracks_full_precision() {
    let (hq, hkv, d) = (4usize, 2usize, 64usize);
    let mut exact = KvCache::new(16);
    let mut packed = QuantizedKvCache::new(8, 32).unwrap();
    let mut seed = 10;
    for n in [6usize, 1, 1] {
        seed += 3;
        let k = noise(hkv * n * d, seed);
        let v = noise(hkv * n * d, seed + 1);
        let q = Tensor::<4>::from_data(TensorData::new(noise(hq * n * d, seed + 2), [1, hq, n, d]), &dev());
        let (ek, ev) = exact.update(kv(k.clone(), hkv, n, d), kv(v.clone(), hkv, n, d));
        let (pk, pv) = packed.update(kv(k, hkv, n, d), kv(v, hkv, n, d));
        let want = read(cache_attention(q.clone(), ek, ev, None));
        let got = read(cache_attention(q, pk, pv, None));
        let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 2e-2, "n={n}: 8-bit cache drifts {err} from full precision");
    }
}

/// `attend` reads the packed cache in place; dense attention over what
/// `update` decodes from an identical cache is the same computation done
/// the long way. Grouped heads, a prefill block, single decode steps and a
/// two-token step (causal mask, still in-place sized), with spare capacity.
#[test]
fn attending_in_place_matches_attention_over_the_decoded_cache() {
    let (hq, hkv, d) = (8usize, 2usize, 64usize);
    for bits in [4u32, 8] {
        let mut decoded = QuantizedKvCache::new(bits, 32).unwrap().with_step(4);
        let mut in_place = QuantizedKvCache::new(bits, 32).unwrap().with_step(4);
        let mut seed = 50;
        for n in [6usize, 1, 2, 1, 3] {
            seed += 3;
            let k = noise(hkv * n * d, seed);
            let v = noise(hkv * n * d, seed + 1);
            let q = Tensor::<4>::from_data(TensorData::new(noise(hq * n * d, seed + 2), [1, hq, n, d]), &dev());
            let (dk, dv) = decoded.update(kv(k.clone(), hkv, n, d), kv(v.clone(), hkv, n, d));
            let want = read(cache_attention(q.clone(), dk, dv, None));
            let got = in_place.attend(q, kv(k, hkv, n, d), kv(v, hkv, n, d), None);
            assert_eq!(got.dims(), [1, hq, n, d]);
            let got = read(got);
            let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-4, "{bits}-bit, n={n} at {}: attend differs by {err}", in_place.offset());
        }
        assert_eq!(in_place.offset(), decoded.offset());
    }
}

#[test]
fn unsupported_widths_are_refused() {
    assert!(QuantizedKvCache::new(3, 32).is_err(), "3-bit codes straddle words");
    assert!(QuantizedKvCache::new(8, 12).is_err(), "groups are multiples of 8");
}
