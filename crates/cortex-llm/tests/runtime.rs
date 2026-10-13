//! KV caches, cache attention and the sampler on the device, on whichever
//! backend this test binary is built for.

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::sampler::CANDIDATES;
use cortex_llm::{KvCache, RotatingKvCache, Sampler, SamplerConfig, cache_attention};

fn dev() -> Device {
    Device::default()
}

fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    match t.try_into_data() {
        Ok(d) => d.try_to_vec::<f32>().unwrap(),
        Err(e) => panic!("device execution failed: {e:#?}"),
    }
}

/// `[1, 1, n, 2]` whose row `i` is `(start + i, -(start + i))`, so every
/// position is identifiable after it has moved through a cache.
fn positions(start: usize, n: usize) -> Tensor<4> {
    let data: Vec<f32> = (0..n)
        .flat_map(|i| {
            let p = (start + i) as f32;
            [p, -p]
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, 1, n, 2]), &dev())
}

fn held(t: Tensor<4>) -> Vec<f32> {
    read(t).chunks(2).map(|r| r[0]).collect()
}

#[test]
fn kv_cache_returns_every_position_in_order_across_growth() {
    let mut cache = KvCache::new(4);
    let mut written = 0;
    for n in [3usize, 3, 1, 5] {
        let (k, v) = cache.update(positions(written, n), positions(written, n));
        written += n;
        let expected: Vec<f32> = (0..written).map(|p| p as f32).collect();
        assert_eq!(held(k), expected, "keys after writing {written}");
        assert_eq!(held(v), expected, "values after writing {written}");
        assert_eq!(cache.offset(), written);
    }
}

#[test]
fn kv_cache_trim_lets_new_tokens_overwrite_rejected_ones() {
    let mut cache = KvCache::new(8);
    cache.update(positions(0, 5), positions(0, 5));
    assert_eq!(cache.trim(2), 2);
    let (k, _) = cache.update(positions(100, 1), positions(100, 1));
    assert_eq!(held(k), vec![0.0, 1.0, 2.0, 100.0]);

    let (sk, sv) = cache.state().unwrap();
    let mut restored = KvCache::from_state(sk, sv, 8);
    let (k, _) = restored.update(positions(7, 1), positions(7, 1));
    assert_eq!(held(k), vec![0.0, 1.0, 2.0, 100.0, 7.0]);
}

#[test]
fn rotating_cache_keeps_the_sinks_and_the_most_recent() {
    // Window 6 keeping 2: after any number of single-token writes the window
    // holds positions {0, 1} and the 4 most recent.
    let mut cache = RotatingKvCache::new(6, 2);
    for p in 0..15usize {
        let (k, _) = cache.update(positions(p, 1), positions(p, 1));
        let mut got = held(k);
        got.sort_by(f32::total_cmp);
        let mut want: Vec<f32> = (0..=p).map(|x| x as f32).collect();
        if want.len() > 6 {
            let recent = want.split_off(want.len() - 4);
            want.truncate(2);
            want.extend(recent);
        }
        assert_eq!(got, want, "window after position {p}");
    }

    // A multi-token write returns everything new, then trims the window.
    let mut cache = RotatingKvCache::new(6, 2);
    cache.update(positions(0, 5), positions(0, 5));
    let (all, _) = cache.update(positions(5, 4), positions(5, 4));
    assert_eq!(held(all), (0..9).map(|x| x as f32).collect::<Vec<_>>());
    let (k, _) = cache.update(positions(9, 1), positions(9, 1));
    let mut got = held(k);
    got.sort_by(f32::total_cmp);
    assert_eq!(got, vec![0.0, 1.0, 6.0, 7.0, 8.0, 9.0]);
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

/// Head counts and lengths of one attention call.
struct Dims {
    hq: usize,
    hkv: usize,
    n: usize,
    len: usize,
    d: usize,
}

/// Host attention for one batch: `q[hq, n, d]`, `k`/`v[hkv, len, d]`, the
/// queries being the last `n` positions.
fn host_attention(q: &[f32], k: &[f32], v: &[f32], dims: Dims) -> Vec<f32> {
    let Dims { hq, hkv, n, len, d } = dims;
    let rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; hq * n * d];
    for h in 0..hq {
        let kh = h / rep;
        for i in 0..n {
            let visible = len - n + i + 1;
            let mut scores: Vec<f32> = (0..visible)
                .map(|j| (0..d).map(|x| q[(h * n + i) * d + x] * k[(kh * len + j) * d + x]).sum::<f32>() * scale)
                .collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                z += *s;
            }
            for j in 0..visible {
                for x in 0..d {
                    out[(h * n + i) * d + x] += scores[j] / z * v[(kh * len + j) * d + x];
                }
            }
        }
    }
    out
}

#[test]
fn cache_attention_matches_the_reference_with_gqa_and_an_offset() {
    let (hq, hkv, d, len) = (4usize, 2usize, 16usize, 7usize);
    for n in [1usize, 3, 7] {
        let q = noise(hq * n * d, 1 + n as u64);
        let k = noise(hkv * len * d, 2);
        let v = noise(hkv * len * d, 3);
        let want = host_attention(&q, &k, &v, Dims { hq, hkv, n, len, d });
        let qt = Tensor::<4>::from_data(TensorData::new(q, [1, hq, n, d]), &dev());
        let kt = Tensor::<4>::from_data(TensorData::new(k, [1, hkv, len, d]), &dev());
        let vt = Tensor::<4>::from_data(TensorData::new(v, [1, hkv, len, d]), &dev());
        let got = read(cache_attention(qt, kt, vt, None));
        let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "n {n}: max error {err}");
    }
}

fn logits(values: &[f32]) -> Tensor<1> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), &dev())
}

#[test]
fn greedy_picks_the_argmax_and_seeds_reproduce() {
    let mut greedy = Sampler::new(SamplerConfig::default());
    assert_eq!(greedy.sample(logits(&[0.1, 2.0, -1.0, 1.9])), 1);

    let cfg = SamplerConfig { temperature: 1.0, seed: 42, ..Default::default() };
    let draw = |cfg: &SamplerConfig| {
        let mut s = Sampler::new(cfg.clone());
        (0..32).map(|_| s.sample(logits(&[0.0, 0.1, 0.2, 0.3]))).collect::<Vec<_>>()
    };
    assert_eq!(draw(&cfg), draw(&cfg), "same seed, same draws");
    let other = SamplerConfig { seed: 43, ..cfg.clone() };
    assert_ne!(draw(&cfg), draw(&other), "different seeds should differ over 32 draws");
}

#[test]
fn distribution_is_exact_and_truncations_cut_where_they_should() {
    let base = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, ..Default::default() });
    let d = s.distribution(logits(&base));
    for (t, p) in [(0u32, 0.5), (1, 0.3), (2, 0.2)] {
        assert!((d.prob(t) - p).abs() < 1e-5, "token {t}: {} vs {p}", d.prob(t));
    }

    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_p: 0.6, ..Default::default() });
    let d = s.distribution(logits(&base));
    assert!((d.prob(0) - 0.625).abs() < 1e-5 && (d.prob(1) - 0.375).abs() < 1e-5 && d.prob(2) == 0.0);

    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, min_p: 0.5, ..Default::default() });
    let d = s.distribution(logits(&base));
    assert_eq!(d.prob(2), 0.0, "0.2 is below half of 0.5");
    assert!(d.prob(1) > 0.0);

    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 1, ..Default::default() });
    assert_eq!(s.distribution(logits(&base)).prob(0), 1.0);
}

#[test]
fn penalties_change_the_choice() {
    let raw = [2.0f32, 1.5, 0.0];
    let mut s = Sampler::new(SamplerConfig { repetition_penalty: 2.0, ..Default::default() });
    assert_eq!(s.sample(logits(&raw)), 0, "nothing seen yet");
    // Token 0 was just emitted: 2.0 / 2 = 1.0 now loses to 1.5.
    assert_eq!(s.sample(logits(&raw)), 1);

    let mut s = Sampler::new(SamplerConfig { presence_penalty: 1.0, ..Default::default() });
    s.observe(0);
    assert_eq!(s.sample(logits(&raw)), 1, "2.0 - 1.0 loses to 1.5");
}

#[test]
fn the_tail_is_read_when_the_candidates_do_not_cover_it() {
    // A flat distribution over more tokens than are read back as candidates:
    // without the full-read fallback, tokens past the candidates would be
    // impossible to draw.
    let vocab = CANDIDATES * 3;
    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, ..Default::default() });
    let d = s.distribution(logits(&vec![0.0f32; vocab]));
    assert_eq!(d.entries().count(), vocab);
    let p = 1.0 / vocab as f64;
    assert!(d.entries().all(|(_, q)| (q - p).abs() < 1e-9));
}

#[test]
fn a_top_k_wider_than_the_candidates_is_ranked_exactly() {
    // Strictly increasing logits: the top `k` are the last `k` tokens, with
    // probabilities proportional to exp(logit).
    let vocab = CANDIDATES * 3;
    let k = CANDIDATES + 10;
    let raw: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.01).collect();
    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: k, ..Default::default() });
    let d = s.distribution(logits(&raw));

    let mut kept: Vec<u32> = d.entries().map(|(t, _)| t).collect();
    kept.sort_unstable();
    let want: Vec<u32> = ((vocab - k) as u32..vocab as u32).collect();
    assert_eq!(kept, want, "exactly the {k} largest logits survive");
    let z: f64 = want.iter().map(|&t| (raw[t as usize] as f64).exp()).sum();
    for t in want {
        let p = (raw[t as usize] as f64).exp() / z;
        assert!((d.prob(t) - p).abs() < 1e-6, "token {t}: {} vs {p}", d.prob(t));
    }
}
