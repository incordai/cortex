//! Fixtures shared by the integration tests (`http.rs`, `hf.rs`).

use std::path::Path;

use safetensors::{Dtype, tensor::TensorView};
use serde_json::json;

/// Writes a small random Llama checkpoint with `vocab` outputs to `dir`,
/// in the Hugging Face layout (`model.safetensors` and `config.json`).
pub fn write_llama(dir: &Path, vocab: usize) {
    let (h, inter, heads, kv, hd) = (64usize, 96usize, 4usize, 2usize, 16usize);
    let mut shapes: Vec<(String, Vec<usize>, f32)> = vec![("model.embed_tokens.weight".into(), vec![vocab, h], 0.0)];
    for i in 0..2 {
        let p = format!("model.layers.{i}");
        shapes.extend([
            (format!("{p}.input_layernorm.weight"), vec![h], 1.0),
            (format!("{p}.self_attn.q_proj.weight"), vec![heads * hd, h], 0.0),
            (format!("{p}.self_attn.k_proj.weight"), vec![kv * hd, h], 0.0),
            (format!("{p}.self_attn.v_proj.weight"), vec![kv * hd, h], 0.0),
            (format!("{p}.self_attn.o_proj.weight"), vec![h, heads * hd], 0.0),
            (format!("{p}.post_attention_layernorm.weight"), vec![h], 1.0),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, h], 0.0),
            (format!("{p}.mlp.up_proj.weight"), vec![inter, h], 0.0),
            (format!("{p}.mlp.down_proj.weight"), vec![h, inter], 0.0),
        ]);
    }
    shapes.push(("model.norm.weight".into(), vec![h], 1.0));
    shapes.push(("lm_head.weight".into(), vec![vocab, h], 0.0));

    let mut seed = 0x5EEDu64;
    let bytes: Vec<Vec<u8>> = shapes
        .iter()
        .map(|(_, shape, offset)| {
            (0..shape.iter().product::<usize>())
                .flat_map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let v = ((seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.4 + offset;
                    v.to_le_bytes()
                })
                .collect()
        })
        .collect();
    let views: Vec<(&str, TensorView<'_>)> = shapes
        .iter()
        .zip(&bytes)
        .map(|((name, shape, _), b)| (name.as_str(), TensorView::new(Dtype::F32, shape.clone(), b).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    let config = json!({
        "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 2,
        "num_attention_heads": heads, "num_key_value_heads": kv, "vocab_size": vocab,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

/// A fresh, empty directory under the system temp dir.
pub fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cortex-serve-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
