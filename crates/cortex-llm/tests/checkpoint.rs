//! Checkpoint loading and conversion, end to end: write a small sharded
//! checkpoint, quantize it, load the packed weights on the device and check
//! the matmul against the original full-precision weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::checkpoint::{Checkpoint, QuantRecipe, quantize_checkpoint};
use cortex_llm::quant::{QuantFormat, QuantMode};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.05
        })
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cortex-llm-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Two shards plus an index, like a downloaded multi-file checkpoint.
fn write_sharded(dir: &Path, q: &[f32], down: &[f32], norm: &[f32]) {
    let a = f32_bytes(q);
    let b = f32_bytes(down);
    let c = f32_bytes(norm);
    let shard1 = vec![
        ("layers.0.q_proj.weight", TensorView::new(Dtype::F32, vec![64, 96], &a).unwrap()),
        ("layers.0.norm.weight", TensorView::new(Dtype::F32, vec![96], &c).unwrap()),
    ];
    let shard2 = vec![("layers.0.down_proj.weight", TensorView::new(Dtype::F32, vec![96, 64], &b).unwrap())];
    safetensors::serialize_to_file(shard1, None, &dir.join("model-00001-of-00002.safetensors")).unwrap();
    safetensors::serialize_to_file(shard2, None, &dir.join("model-00002-of-00002.safetensors")).unwrap();
    let mut map = HashMap::new();
    map.insert("layers.0.q_proj.weight", "model-00001-of-00002.safetensors");
    map.insert("layers.0.norm.weight", "model-00001-of-00002.safetensors");
    map.insert("layers.0.down_proj.weight", "model-00002-of-00002.safetensors");
    let index = format!(
        "{{\"weight_map\": {{{}}}}}",
        map.iter().map(|(k, v)| format!("\"{k}\": \"{v}\"")).collect::<Vec<_>>().join(", ")
    );
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
}

#[test]
fn sharded_checkpoints_load_by_name() {
    let dir = scratch("load");
    let (q, down, norm) = (noise(64 * 96, 1), noise(96 * 64, 2), noise(96, 3));
    write_sharded(&dir, &q, &down, &norm);

    let ckpt = Checkpoint::open(&dir).unwrap();
    assert_eq!(
        ckpt.names(),
        vec!["layers.0.down_proj.weight", "layers.0.norm.weight", "layers.0.q_proj.weight"]
    );
    let (shape, values) = ckpt.read_f32("layers.0.down_proj.weight").unwrap();
    assert_eq!(shape, vec![96, 64]);
    assert_eq!(values, down);
    let t = ckpt.tensor::<1>("layers.0.norm.weight", &Device::default()).unwrap();
    assert_eq!(t.into_data().try_to_vec::<f32>().unwrap(), norm);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tensor_data_shares_the_mapped_file_instead_of_copying() {
    let dir = scratch("zerocopy");
    let (q, down, norm) = (noise(64 * 96, 1), noise(96 * 64, 2), noise(96, 3));
    write_sharded(&dir, &q, &down, &norm);
    let ckpt = Checkpoint::open(&dir).unwrap();
    for name in ["layers.0.q_proj.weight", "layers.0.down_proj.weight", "layers.0.norm.weight"] {
        let mapped = ckpt.view(name).unwrap().data().as_ptr();
        let data = ckpt.tensor_data(name).unwrap();
        assert_eq!(data.as_bytes().as_ptr(), mapped, "{name} was copied out of the map");
    }
    let values = ckpt.tensor_data("layers.0.norm.weight").unwrap().try_to_vec::<f32>().unwrap();
    assert_eq!(values, norm);
    drop(ckpt);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_index_cannot_point_outside_the_checkpoint() {
    let dir = scratch("traversal");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        "{\"weight_map\": {\"w\": \"../secret.safetensors\"}}",
    )
    .unwrap();
    let err = Checkpoint::open(&dir).err().expect("a traversing shard name must be refused");
    assert!(err.to_string().contains("not a plain file name"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn quantized_checkpoints_reproduce_the_original_matmul() {
    let dir = scratch("convert-in");
    let out = scratch("convert-out");
    let (q, down, norm) = (noise(64 * 96, 1), noise(96 * 64, 2), noise(96, 3));
    write_sharded(&dir, &q, &down, &norm);

    // q_proj at 4 bits; down_proj kept at 8 by the mixed recipe.
    let low = QuantFormat::new(4, 32, QuantMode::Affine).unwrap();
    let high = QuantFormat::new(8, 32, QuantMode::Affine).unwrap();
    let report = quantize_checkpoint(&dir, &out, &QuantRecipe::mixed(low, high)).unwrap();
    assert_eq!((report.quantized, report.kept), (2, 1), "both projections packed, the norm kept");

    let ckpt = Checkpoint::open(&out).unwrap();
    let formats = ckpt.quant_formats().unwrap();
    assert_eq!(formats["layers.0.q_proj.weight"].bits(), 4);
    assert_eq!(formats["layers.0.down_proj.weight"].bits(), 8);
    assert!(ckpt.contains("layers.0.norm.weight"));

    let dev = Device::default();
    for (name, w, out_f, in_f, tol) in [
        ("layers.0.q_proj.weight", &q, 64usize, 96usize, 0.05f32),
        ("layers.0.down_proj.weight", &down, 96, 64, 0.005),
    ] {
        let mat = ckpt.quantized_matrix(name, &dev).unwrap();
        let x = noise(2 * in_f, 9);
        // Reference: x · Wᵀ with W stored [out, in].
        let mut want = vec![0f32; 2 * out_f];
        for r in 0..2 {
            for o in 0..out_f {
                want[r * out_f + o] = (0..in_f).map(|i| x[r * in_f + i] * w[o * in_f + i]).sum();
            }
        }
        let xt = Tensor::<2>::from_data(TensorData::new(x, [2, in_f]), &dev);
        let got = mat.forward(xt).into_data().try_to_vec::<f32>().unwrap();
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err <= tol * scale.max(1.0), "{name}: max error {err} (scale {scale})");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&out);
}
