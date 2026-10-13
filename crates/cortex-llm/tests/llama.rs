//! The Llama-family model end to end on whichever backend this test binary
//! is built for: a small random checkpoint written to disk in the Hugging
//! Face layout, loaded through `Llama::load`, and checked against an
//! independent host forward pass; then incremental decoding, batched decoding
//! and a quantized copy of the checkpoint against the dense model.

use std::path::{Path, PathBuf};

use cortex_core::tensor::{Device, Tensor};
use cortex_llm::checkpoint::{QuantRecipe, quantize_checkpoint};
use cortex_llm::llama::{Llama, LlamaConfig};
use cortex_llm::quant::{QuantFormat, QuantMode};
use cortex_llm::{LanguageModel, rms_norm_reference, rope_reference};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

const HIDDEN: usize = 64;
const INTER: usize = 96;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 80;
const EPS: f32 = 1e-5;
const THETA: f32 = 10_000.0;

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

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cortex-llm-llama-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every weight of the test model by its checkpoint name, with its shape.
fn weights() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut w = Vec::new();
    let mut seed = 0;
    let mut add = |name: String, shape: Vec<usize>, amp: f32, offset: f32| {
        seed += 1;
        let len = shape.iter().product();
        let values = noise(len, seed, amp).into_iter().map(|v| v + offset).collect();
        w.push((name, shape, values));
    };
    add("model.embed_tokens.weight".into(), vec![VOCAB, HIDDEN], 1.0, 0.0);
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        add(format!("{p}.input_layernorm.weight"), vec![HIDDEN], 0.2, 1.0);
        add(format!("{p}.self_attn.q_proj.weight"), vec![HEADS * HEAD_DIM, HIDDEN], 0.25, 0.0);
        add(format!("{p}.self_attn.k_proj.weight"), vec![KV_HEADS * HEAD_DIM, HIDDEN], 0.25, 0.0);
        add(format!("{p}.self_attn.v_proj.weight"), vec![KV_HEADS * HEAD_DIM, HIDDEN], 0.25, 0.0);
        add(format!("{p}.self_attn.o_proj.weight"), vec![HIDDEN, HEADS * HEAD_DIM], 0.25, 0.0);
        add(format!("{p}.post_attention_layernorm.weight"), vec![HIDDEN], 0.2, 1.0);
        add(format!("{p}.mlp.gate_proj.weight"), vec![INTER, HIDDEN], 0.25, 0.0);
        add(format!("{p}.mlp.up_proj.weight"), vec![INTER, HIDDEN], 0.25, 0.0);
        add(format!("{p}.mlp.down_proj.weight"), vec![HIDDEN, INTER], 0.25, 0.0);
    }
    add("model.norm.weight".into(), vec![HIDDEN], 0.2, 1.0);
    add("lm_head.weight".into(), vec![VOCAB, HIDDEN], 0.25, 0.0);
    w
}

const CONFIG: &str = r#"{
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 80,
    "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false
}"#;

fn write_checkpoint(dir: &Path) {
    let w = weights();
    let bytes: Vec<Vec<u8>> = w.iter().map(|(_, _, v)| v.iter().flat_map(|x| x.to_le_bytes()).collect()).collect();
    let views: Vec<(&str, TensorView<'_>)> = w
        .iter()
        .zip(&bytes)
        .map(|((name, shape, _), b)| (name.as_str(), TensorView::new(Dtype::F32, shape.clone(), b).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
}

fn read(t: Tensor<2>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// `x[n, in] · W[out, in]ᵀ` on the host.
fn linear(x: &[f32], w: &[f32], n: usize, input: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * out];
    for i in 0..n {
        for o in 0..out {
            y[i * out + o] = (0..input).map(|c| x[i * input + c] * w[o * input + c]).sum();
        }
    }
    y
}

/// `[n, heads·hd]` → `[heads, n, hd]`.
fn split_heads(x: &[f32], n: usize, heads: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..HEAD_DIM {
                out[(h * n + t) * HEAD_DIM + d] = x[t * heads * HEAD_DIM + h * HEAD_DIM + d];
            }
        }
    }
    out
}

/// The whole model on the host, from the checkpoint's own numbers: one
/// sequence, every position at once, written from the Llama definition
/// rather than from the device code.
fn host_forward(tokens: &[u32]) -> Vec<f32> {
    let w: std::collections::HashMap<String, Vec<f32>> = weights().into_iter().map(|(n, _, v)| (n, v)).collect();
    let n = tokens.len();
    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|p| THETA.powf(-2.0 * p as f32 / HEAD_DIM as f32))
        .collect();
    let mut x: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| w["model.embed_tokens.weight"][t as usize * HIDDEN..(t as usize + 1) * HIDDEN].to_vec())
        .collect();

    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        let h = rms_norm_reference(&x, &w[&format!("{p}.input_layernorm.weight")], EPS, false);
        let q = linear(&h, &w[&format!("{p}.self_attn.q_proj.weight")], n, HIDDEN, HEADS * HEAD_DIM);
        let k = linear(&h, &w[&format!("{p}.self_attn.k_proj.weight")], n, HIDDEN, KV_HEADS * HEAD_DIM);
        let v = linear(&h, &w[&format!("{p}.self_attn.v_proj.weight")], n, HIDDEN, KV_HEADS * HEAD_DIM);
        let q = rope_reference(&split_heads(&q, n, HEADS), n, HEAD_DIM, &inv_freq, 0, false);
        let k = rope_reference(&split_heads(&k, n, KV_HEADS), n, HEAD_DIM, &inv_freq, 0, false);
        let v = split_heads(&v, n, KV_HEADS);

        // Causal softmax attention, query head `h` reading kv head `h / rep`.
        let rep = HEADS / KV_HEADS;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn = vec![0f32; n * HEADS * HEAD_DIM];
        for head in 0..HEADS {
            let kv = head / rep;
            for t in 0..n {
                let qrow = &q[(head * n + t) * HEAD_DIM..][..HEAD_DIM];
                let scores: Vec<f32> = (0..=t)
                    .map(|s| {
                        let krow = &k[(kv * n + s) * HEAD_DIM..][..HEAD_DIM];
                        qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let top = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f32 = exp.iter().sum();
                for d in 0..HEAD_DIM {
                    let value: f32 = (0..=t).map(|s| exp[s] / total * v[(kv * n + s) * HEAD_DIM + d]).sum();
                    attn[t * HEADS * HEAD_DIM + head * HEAD_DIM + d] = value;
                }
            }
        }
        let o = linear(&attn, &w[&format!("{p}.self_attn.o_proj.weight")], n, HEADS * HEAD_DIM, HIDDEN);
        x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

        let h = rms_norm_reference(&x, &w[&format!("{p}.post_attention_layernorm.weight")], EPS, false);
        let gate = linear(&h, &w[&format!("{p}.mlp.gate_proj.weight")], n, HIDDEN, INTER);
        let up = linear(&h, &w[&format!("{p}.mlp.up_proj.weight")], n, HIDDEN, INTER);
        let mlp: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
        let down = linear(&mlp, &w[&format!("{p}.mlp.down_proj.weight")], n, INTER, HIDDEN);
        x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
    }
    let x = rms_norm_reference(&x, &w["model.norm.weight"], EPS, false);
    linear(&x, &w["lm_head.weight"], n, HIDDEN, VOCAB)
}

fn model(name: &str) -> (Llama, PathBuf) {
    let dir = scratch(name);
    write_checkpoint(&dir);
    (Llama::load(&dir, &Device::default()).unwrap(), dir)
}

const PROMPT: [u32; 7] = [3, 17, 42, 5, 79, 0, 23];

#[test]
fn a_prompt_matches_an_independent_host_forward_pass() {
    let (llama, dir) = model("host");
    let mut state = llama.new_state();
    let got = read(llama.forward(&mut state, &PROMPT));
    let want = host_forward(&PROMPT);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = max_diff(&got, &want);
    assert!(err < 1e-4 * scale.max(1.0), "logits differ by {err} (scale {scale})");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn decoding_token_by_token_matches_one_prompt() {
    let (llama, dir) = model("incremental");
    let whole = read(llama.forward(&mut llama.new_state(), &PROMPT));

    let mut session = llama.session();
    let mut pieces = Vec::new();
    for chunk in [&PROMPT[..3], &PROMPT[3..4], &PROMPT[4..5], &PROMPT[5..]] {
        pieces.extend(read(session.forward(chunk)));
    }
    let err = max_diff(&pieces, &whole);
    assert!(err < 1e-4, "incremental decoding drifts by {err}");

    // Trimming forgets the last positions: feeding them again gives the
    // same logits.
    session.trim(2);
    let again = read(session.forward(&PROMPT[5..]));
    assert!(max_diff(&again, &whole[5 * VOCAB..]) < 1e-4);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_batched_step_matches_each_sequence_alone() {
    let (llama, dir) = model("batch");
    let prompts: [&[u32]; 3] = [&[1, 2], &[9, 8, 7, 6, 5], &[44, 33, 22]];
    let mut batched: Vec<_> = prompts.iter().map(|_| llama.new_state()).collect();
    let mut alone: Vec<_> = prompts.iter().map(|_| llama.new_state()).collect();
    for (i, p) in prompts.iter().enumerate() {
        llama.forward(&mut batched[i], p);
        llama.forward(&mut alone[i], p);
    }

    // Two decode steps; the sequences sit at three different lengths.
    for step in [[10u32, 11, 12], [70, 0, 79]] {
        let mut refs: Vec<&mut _> = batched.iter_mut().collect();
        let got = read(llama.forward_batch(&mut refs, &step));
        assert_eq!(got.len(), 3 * VOCAB);
        for (i, &token) in step.iter().enumerate() {
            let want = read(llama.forward(&mut alone[i], &[token]));
            let err = max_diff(&got[i * VOCAB..(i + 1) * VOCAB], &want);
            assert!(err < 1e-4, "sequence {i} in the batch differs by {err}");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_quantized_checkpoint_loads_packed_and_tracks_the_dense_model() {
    let (dense, dir) = model("dense-for-quant");
    let packed_dir = scratch("packed");
    let format = QuantFormat::new(8, 32, QuantMode::Affine).unwrap();
    let report = quantize_checkpoint(dir.join("model.safetensors"), &packed_dir, &QuantRecipe::uniform(format)).unwrap();
    // Every projection whose output width is a multiple of 32: q, k, v, o,
    // gate, up and down in both layers. The head and embeddings (80 wide)
    // stay dense.
    assert_eq!(report.quantized, 7 * LAYERS);
    std::fs::write(packed_dir.join("config.json"), CONFIG).unwrap();
    let packed = Llama::load(&packed_dir, &Device::default()).unwrap();

    let want = read(dense.forward(&mut dense.new_state(), &PROMPT));
    let got = read(packed.forward(&mut packed.new_state(), &PROMPT));
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = max_diff(&got, &want);
    assert!(err < 0.05 * scale, "8-bit model drifts {err} from the dense one (scale {scale})");
    assert!(err > 0.0, "the packed weights were not used");
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(packed_dir);
}

#[test]
fn inconsistent_configs_are_refused() {
    let bad_heads = CONFIG.replace("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3");
    assert!(LlamaConfig::from_json(&bad_heads).is_err());
    let missing = CONFIG.replace("\"vocab_size\": 80,", "");
    assert!(LlamaConfig::from_json(&missing).is_err());
    let config = LlamaConfig::from_json(CONFIG).unwrap();
    assert_eq!((config.head_dim, config.num_key_value_heads), (HEAD_DIM, KV_HEADS));
}
