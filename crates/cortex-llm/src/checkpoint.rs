//! Loading safetensors checkpoints (single file or sharded with an index),
//! memory-mapped and read lazily by tensor name, and converting them to the
//! packed quantized format.
//!
//! Callers: model loaders and the `cortex-quantize` tool. Calls: `quant.rs`
//! (packing) and `quantized.rs` (`QuantizedMatrix::from_tensors`).
//! Sibling: `cortex-store`, which loads whole `Module`s rather than tensors
//! by name.
//!
//! Files are untrusted input: every tensor's byte range is checked against
//! its shard, and shard names from an index must be plain file names.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use cortex_core::tensor::{AllocationProperty, Bytes, DType, Device, Int, Tensor, TensorData};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::tensor::{Metadata, SafeTensors, TensorView};

use crate::QuantizedMatrix;
use crate::quant::{QuantError, QuantFormat, QuantMode, quantize};

/// Metadata key holding the per-tensor quantization formats as JSON.
pub const QUANT_METADATA_KEY: &str = "cortex.quantization";

/// Why a checkpoint could not be read or written.
#[derive(Debug)]
pub enum CheckpointError {
    /// A file could not be opened, mapped or written.
    Io(PathBuf, std::io::Error),
    /// A shard or index is not valid.
    Format(String),
    /// No tensor with this name.
    Missing(String),
    /// A tensor has a dtype this operation does not handle.
    Dtype(String, Dtype),
    /// Packing failed.
    Quant(QuantError),
}

impl core::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::Format(m) => write!(f, "invalid checkpoint: {m}"),
            Self::Missing(n) => write!(f, "no tensor named {n}"),
            Self::Dtype(n, d) => write!(f, "tensor {n} has unsupported dtype {d:?}"),
            Self::Quant(e) => write!(f, "quantization failed: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<QuantError> for CheckpointError {
    fn from(e: QuantError) -> Self {
        Self::Quant(e)
    }
}

/// One mapped file. The map is held as shared bytes so tensors can be handed
/// to the device as slices of it, without a host copy.
struct Shard {
    path: PathBuf,
    map: bytes::Bytes,
    header_len: usize,
    meta: Metadata,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, CheckpointError> {
        let file = File::open(path).map_err(|e| CheckpointError::Io(path.into(), e))?;
        // SAFETY: the map is read-only and the file is not modified while the
        // checkpoint is open; a concurrent writer is outside this API's model.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| CheckpointError::Io(path.into(), e))?;
        let (header_len, meta) = SafeTensors::read_metadata(&map)
            .map_err(|e| CheckpointError::Format(format!("{}: {e:?}", path.display())))?;
        let body = map.len().saturating_sub(8 + header_len);
        for (name, info) in meta.tensors() {
            if info.data_offsets.0 > info.data_offsets.1 || info.data_offsets.1 > body {
                return Err(CheckpointError::Format(format!(
                    "{}: tensor {name} lies outside the file",
                    path.display()
                )));
            }
        }
        Ok(Self {
            path: path.into(),
            map: bytes::Bytes::from_owner(map),
            header_len,
            meta,
        })
    }

    /// Byte range of `name` within the file.
    fn range(&self, name: &str) -> Option<core::ops::Range<usize>> {
        let info = self.meta.info(name)?;
        Some(8 + self.header_len + info.data_offsets.0..8 + self.header_len + info.data_offsets.1)
    }

    fn view(&self, name: &str) -> Option<TensorView<'_>> {
        let info = self.meta.info(name)?;
        TensorView::new(info.dtype, info.shape.clone(), &self.map[self.range(name)?]).ok()
    }

    /// `name`'s bytes as a slice sharing the map (no copy).
    fn shared(&self, name: &str) -> Option<bytes::Bytes> {
        Some(self.map.slice(self.range(name)?))
    }
}

/// An open checkpoint: one or more memory-mapped shards and a name index.
pub struct Checkpoint {
    shards: Vec<Shard>,
    index: HashMap<String, usize>,
}

fn plain_file_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(['/', '\\', ':'])
        && name != "."
        && name != ".."
}

impl Checkpoint {
    /// Opens a `.safetensors` file, or a directory holding either
    /// `model.safetensors.index.json` (sharded) or `.safetensors` files.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let path = path.as_ref();
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            let index = path.join("model.safetensors.index.json");
            if index.is_file() {
                let text = std::fs::read_to_string(&index).map_err(|e| CheckpointError::Io(index.clone(), e))?;
                let json: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| CheckpointError::Format(format!("{}: {e}", index.display())))?;
                let map = json
                    .get("weight_map")
                    .and_then(|m| m.as_object())
                    .ok_or_else(|| CheckpointError::Format("index has no weight_map".into()))?;
                let mut names: Vec<String> = Vec::new();
                for shard in map.values() {
                    let shard = shard
                        .as_str()
                        .ok_or_else(|| CheckpointError::Format("weight_map value is not a string".into()))?;
                    if !plain_file_name(shard) {
                        return Err(CheckpointError::Format(format!(
                            "shard name {shard:?} is not a plain file name"
                        )));
                    }
                    if !names.iter().any(|n| n == shard) {
                        names.push(shard.to_string());
                    }
                }
                names.sort();
                names.into_iter().map(|n| path.join(n)).collect()
            } else {
                let mut found: Vec<PathBuf> = std::fs::read_dir(path)
                    .map_err(|e| CheckpointError::Io(path.into(), e))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                    .collect();
                found.sort();
                found
            }
        };
        if files.is_empty() {
            return Err(CheckpointError::Format(format!("no safetensors in {}", path.display())));
        }

        let mut shards = Vec::with_capacity(files.len());
        let mut index = HashMap::new();
        for file in &files {
            let shard = Shard::open(file)?;
            for name in shard.meta.tensors().keys() {
                if index.insert(name.clone(), shards.len()).is_some() {
                    return Err(CheckpointError::Format(format!("tensor {name} appears in two shards")));
                }
            }
            shards.push(shard);
        }
        Ok(Self { shards, index })
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut n: Vec<&str> = self.index.keys().map(String::as_str).collect();
        n.sort_unstable();
        n
    }

    /// Whether a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// The raw view of a tensor (zero-copy into the mapped file).
    pub fn view(&self, name: &str) -> Result<TensorView<'_>, CheckpointError> {
        let shard = self.index.get(name).ok_or_else(|| CheckpointError::Missing(name.into()))?;
        self.shards[*shard]
            .view(name)
            .ok_or_else(|| CheckpointError::Format(format!("{name} has an inconsistent shape")))
    }

    /// The shard file holding `name`.
    pub fn shard_of(&self, name: &str) -> Option<&Path> {
        self.index.get(name).map(|s| self.shards[*s].path.as_path())
    }

    /// The checkpoint's free-form metadata (merged across shards).
    pub fn metadata(&self, key: &str) -> Option<String> {
        self.shards
            .iter()
            .find_map(|s| s.meta.metadata().as_ref().and_then(|m| m.get(key).cloned()))
    }

    /// A floating-point tensor converted to `f32`, with its shape.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), CheckpointError> {
        let view = self.view(name)?;
        let bytes = view.data();
        let values = match view.dtype() {
            Dtype::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            other => return Err(CheckpointError::Dtype(name.into(), other)),
        };
        Ok((view.shape().to_vec(), values))
    }

    /// The host data of `name` as stored (f32 / f16 / bf16 / u32), sharing
    /// the memory-mapped file: no bytes are copied until a device upload.
    ///
    /// Callers: [`Checkpoint::tensor`], [`Checkpoint::quantized_matrix`].
    pub fn tensor_data(&self, name: &str) -> Result<TensorData, CheckpointError> {
        let shard = &self.shards[*self.index.get(name).ok_or_else(|| CheckpointError::Missing(name.into()))?];
        let view = shard
            .view(name)
            .ok_or_else(|| CheckpointError::Format(format!("{name} has an inconsistent shape")))?;
        let dtype = match view.dtype() {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::U32 => DType::U32,
            other => return Err(CheckpointError::Dtype(name.into(), other)),
        };
        let shape = view.shape().to_vec();
        let bytes = shard.shared(name).expect("the view above resolved this name");
        // `File`: the bytes live in a mapped file, which GPU runtimes stage
        // through pinned memory.
        let bytes = Bytes::from_shared(bytes, AllocationProperty::File);
        TensorData::try_from_bytes(bytes, shape, dtype).map_err(|e| CheckpointError::Format(format!("{name}: {e}")))
    }

    /// A floating-point tensor on `device`, keeping f16/bf16/f32 storage.
    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>, CheckpointError> {
        let data = self.tensor_data(name)?;
        if data.rank() != D {
            return Err(CheckpointError::Format(format!("{name} has rank {}, expected {D}", data.rank())));
        }
        let dtype = data.dtype();
        if !dtype.is_float() {
            return Err(CheckpointError::Format(format!("{name} is not a floating-point tensor")));
        }
        Ok(Tensor::from_data(data, (device, dtype)))
    }

    /// The packed weight stored under `name` by [`quantize_checkpoint`], on
    /// `device`.
    pub fn quantized_matrix(&self, name: &str, device: &Device) -> Result<QuantizedMatrix, CheckpointError> {
        let formats = self.quant_formats()?;
        let format = *formats
            .get(name)
            .ok_or_else(|| CheckpointError::Missing(format!("{name} (not quantized)")))?;
        let qweight = self.view(&format!("{name}.qweight"))?;
        if qweight.dtype() != Dtype::U32 || qweight.shape().len() != 2 {
            return Err(CheckpointError::Format(format!("{name}.qweight must be a 2-D u32 tensor")));
        }
        let data = Tensor::<2, Int>::from_data(self.tensor_data(&format!("{name}.qweight"))?, (device, DType::U32));
        let scales = self.tensor::<2>(&format!("{name}.scales"), device)?;
        let biases = match format.mode() {
            QuantMode::Affine => Some(self.tensor::<2>(&format!("{name}.biases"), device)?),
            QuantMode::Symmetric => None,
        };
        Ok(QuantizedMatrix::from_tensors(data, scales, biases, format)?)
    }

    /// The quantization formats recorded in the checkpoint, by weight name.
    pub fn quant_formats(&self) -> Result<HashMap<String, QuantFormat>, CheckpointError> {
        let mut out = HashMap::new();
        for shard in &self.shards {
            let Some(json) = shard.meta.metadata().as_ref().and_then(|m| m.get(QUANT_METADATA_KEY)) else {
                continue;
            };
            let value: serde_json::Value = serde_json::from_str(json)
                .map_err(|e| CheckpointError::Format(format!("{QUANT_METADATA_KEY}: {e}")))?;
            let obj = value
                .as_object()
                .ok_or_else(|| CheckpointError::Format(format!("{QUANT_METADATA_KEY} is not an object")))?;
            for (name, f) in obj {
                let bits = f.get("bits").and_then(|v| v.as_u64());
                let group = f.get("group_size").and_then(|v| v.as_u64());
                let mode = match f.get("mode").and_then(|v| v.as_str()) {
                    Some("affine") => Some(QuantMode::Affine),
                    Some("symmetric") => Some(QuantMode::Symmetric),
                    _ => None,
                };
                let (Some(bits), Some(group), Some(mode)) = (bits, group, mode) else {
                    return Err(CheckpointError::Format(format!("bad quantization entry for {name}")));
                };
                out.insert(name.clone(), QuantFormat::new(bits as u32, group as usize, mode)?);
            }
        }
        Ok(out)
    }
}

/// Which weights to quantize, and how.
#[derive(Clone, Debug)]
pub struct QuantRecipe {
    /// Format for every eligible weight not matched by an override.
    pub default: QuantFormat,
    /// `(substring, format)`: a weight whose name contains `substring` uses
    /// `format` (`None` keeps it in full precision). First match wins.
    pub overrides: Vec<(String, Option<QuantFormat>)>,
}

impl QuantRecipe {
    /// Uniform quantization of every eligible weight.
    pub fn uniform(format: QuantFormat) -> Self {
        Self { default: format, overrides: Vec::new() }
    }

    /// Mixed precision: `low` everywhere, `high` for the projections that
    /// are most sensitive to quantization error (value and down projections,
    /// the output head and embeddings).
    pub fn mixed(low: QuantFormat, high: QuantFormat) -> Self {
        let overrides = ["v_proj", "down_proj", "lm_head", "embed_tokens"]
            .iter()
            .map(|s| (s.to_string(), Some(high)))
            .collect();
        Self { default: low, overrides }
    }

    /// The format for `name`, or `None` to keep it as is. Only 2-D weights
    /// named `*.weight` are eligible; norms and biases are always kept.
    pub fn format_for(&self, name: &str, shape: &[usize]) -> Option<QuantFormat> {
        if shape.len() != 2 || !name.ends_with(".weight") || name.contains("norm") {
            return None;
        }
        let chosen = self
            .overrides
            .iter()
            .find(|(pat, _)| name.contains(pat.as_str()))
            .map_or(Some(self.default), |(_, f)| *f)?;
        // Stored as [out, in]; packed as [in, out], so `out` must align.
        chosen.check_columns(shape[0]).ok().map(|_| chosen)
    }
}

/// An owned tensor ready to be written.
struct Owned {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for &Owned {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Summary of a conversion.
#[derive(Clone, Debug, Default)]
pub struct QuantizeReport {
    /// Weights packed.
    pub quantized: usize,
    /// Tensors copied unchanged.
    pub kept: usize,
    /// Bytes written (tensor data only).
    pub bytes: usize,
}

/// Converts the checkpoint at `input` into `output_dir`, packing every weight
/// `recipe` selects. Keeps the input's sharding (one output file per input
/// shard, same names) and writes an index when the input had more than one
/// shard. Packed weights become `<name>.qweight` (u32 `[in, words]`),
/// `<name>.scales` (f16 `[in, groups]`) and, in affine mode,
/// `<name>.biases`; their formats go in each file's metadata under
/// [`QUANT_METADATA_KEY`].
pub fn quantize_checkpoint(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    recipe: &QuantRecipe,
) -> Result<QuantizeReport, CheckpointError> {
    let ckpt = Checkpoint::open(input)?;
    let out = output_dir.as_ref();
    std::fs::create_dir_all(out).map_err(|e| CheckpointError::Io(out.into(), e))?;

    let mut by_shard: HashMap<PathBuf, Vec<&str>> = HashMap::new();
    for name in ckpt.names() {
        let shard = ckpt.shard_of(name).expect("indexed").to_path_buf();
        by_shard.entry(shard).or_default().push(name);
    }
    let mut report = QuantizeReport::default();
    let mut weight_map = serde_json::Map::new();

    let mut shard_paths: Vec<PathBuf> = by_shard.keys().cloned().collect();
    shard_paths.sort();
    for shard in &shard_paths {
        let file_name = shard
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| CheckpointError::Format(format!("{} has no file name", shard.display())))?
            .to_string();
        let mut tensors: Vec<(String, Owned)> = Vec::new();
        let mut formats = serde_json::Map::new();
        for &name in &by_shard[shard] {
            let view = ckpt.view(name)?;
            match recipe.format_for(name, view.shape()) {
                Some(format) => {
                    let (shape, w) = ckpt.read_f32(name)?;
                    let (out_features, in_features) = (shape[0], shape[1]);
                    // [out, in] -> [in, out] so the packed rows are inputs.
                    let mut wt = vec![0f32; w.len()];
                    for o in 0..out_features {
                        for i in 0..in_features {
                            wt[i * out_features + o] = w[o * in_features + i];
                        }
                    }
                    let packed = quantize(&wt, in_features, out_features, format)?;
                    let wpr = format.words_per_row(out_features);
                    let gpr = format.groups_per_row(out_features);
                    let to_f16 = |v: &[f32]| v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect();
                    tensors.push((
                        format!("{name}.qweight"),
                        Owned {
                            dtype: Dtype::U32,
                            shape: vec![in_features, wpr],
                            data: packed.data.iter().flat_map(|x| x.to_le_bytes()).collect(),
                        },
                    ));
                    tensors.push((
                        format!("{name}.scales"),
                        Owned { dtype: Dtype::F16, shape: vec![in_features, gpr], data: to_f16(&packed.scales) },
                    ));
                    if let Some(b) = &packed.biases {
                        tensors.push((
                            format!("{name}.biases"),
                            Owned { dtype: Dtype::F16, shape: vec![in_features, gpr], data: to_f16(b) },
                        ));
                    }
                    let mode = match format.mode() {
                        QuantMode::Affine => "affine",
                        QuantMode::Symmetric => "symmetric",
                    };
                    formats.insert(
                        name.to_string(),
                        serde_json::json!({ "bits": format.bits(), "group_size": format.group_size(), "mode": mode }),
                    );
                    report.quantized += 1;
                }
                None => {
                    tensors.push((
                        name.to_string(),
                        Owned { dtype: view.dtype(), shape: view.shape().to_vec(), data: view.data().to_vec() },
                    ));
                    report.kept += 1;
                }
            }
        }
        report.bytes += tensors.iter().map(|(_, t)| t.data.len()).sum::<usize>();
        for (n, _) in &tensors {
            weight_map.insert(n.clone(), serde_json::Value::String(file_name.clone()));
        }
        let mut info = HashMap::new();
        info.insert(QUANT_METADATA_KEY.to_string(), serde_json::Value::Object(formats).to_string());
        let path = out.join(&file_name);
        safetensors::serialize_to_file(tensors.iter().map(|(n, t)| (n.as_str(), t)), Some(info), &path)
            .map_err(|e| CheckpointError::Format(format!("{}: {e:?}", path.display())))?;
    }

    if shard_paths.len() > 1 {
        let index = serde_json::json!({ "metadata": { "total_size": report.bytes }, "weight_map": weight_map });
        let path = out.join("model.safetensors.index.json");
        std::fs::write(&path, index.to_string()).map_err(|e| CheckpointError::Io(path, e))?;
    }
    Ok(report)
}
