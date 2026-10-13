//! A key/value cache held in the packed quantized format of [`crate::quant`].
//!
//! Each cached position of each head is one row of `head_dim` columns, so the
//! cache is a packed `[positions, head_dim]` matrix: `u32` words of affine
//! codes plus a scale and a bias per group, in the dtype of the incoming
//! keys. At 8 bits that is about a quarter of an f32 cache, at 4 bits about
//! an eighth (plus the per-group scale and bias).
//!
//! Writes quantize only the new positions, with tensor ops on the device.
//! [`QuantizedKvCache::attend`] reads the cache where it lies: the scores
//! and the weighted sum are matmuls against the packed rows
//! ([`LlmOps::batched_quantized_matmul`]), so decode steps never build a
//! full-precision copy. [`QuantizedKvCache::update`] instead decodes every
//! held position through [`LlmOps::dequantize_packed`], for callers that
//! need the keys and values themselves.
//!
//! Callers: model code, in place of [`crate::KvCache`] (same `update` /
//! `trim` / `offset` contract, plus `attend`); `prefix.rs` through
//! [`PrefixState`]. Sibling: `cache.rs` (`KvCache`, full precision, and
//! `cache_attention`, the dense attention `attend` agrees with).

use alloc::vec::Vec;
use cortex_core::backend::Dispatch;
use cortex_core::tensor::activation::softmax;
use cortex_core::tensor::{DType, Int, Tensor};
use cortex_std::IntDType;

use crate::LlmOps;
use crate::cache::{DEFAULT_STEP, causal_mask};
use crate::prefix::PrefixState;
use crate::quant::{QuantError, QuantFormat, QuantMode};

/// One side (keys or values) of the cache: `[b, h, capacity, ·]` buffers.
#[derive(Clone, Debug)]
struct PackedRows {
    data: Tensor<4, Int>,
    scales: Tensor<4>,
    biases: Tensor<4>,
}

impl PackedRows {
    fn capacity(&self) -> usize {
        self.data.dims()[2]
    }

    fn cat(self, other: Self) -> Self {
        Self {
            data: Tensor::cat(alloc::vec![self.data, other.data], 2),
            scales: Tensor::cat(alloc::vec![self.scales, other.scales], 2),
            biases: Tensor::cat(alloc::vec![self.biases, other.biases], 2),
        }
    }

    fn write(self, at: usize, rows: Self) -> Self {
        let [b, h, n, w] = rows.data.dims();
        let g = rows.scales.dims()[3];
        Self {
            data: self.data.slice_assign([0..b, 0..h, at..at + n, 0..w], rows.data),
            scales: self.scales.slice_assign([0..b, 0..h, at..at + n, 0..g], rows.scales),
            biases: self.biases.slice_assign([0..b, 0..h, at..at + n, 0..g], rows.biases),
        }
    }

    fn bytes(&self) -> usize {
        let elems = |dims: [usize; 4]| dims.iter().product::<usize>();
        elems(self.data.dims()) * 4
            + (elems(self.scales.dims()) + elems(self.biases.dims())) * self.scales.dtype().size()
    }
}

/// A growable key/value cache stored at `bits` per value.
///
/// Keys and values are `[batch, kv_heads, positions, head_dim]`; `head_dim`
/// must be a multiple of 32 and of the group size.
#[derive(Clone, Debug)]
pub struct QuantizedKvCache {
    format: QuantFormat,
    keys: Option<PackedRows>,
    values: Option<PackedRows>,
    offset: usize,
    step: usize,
}

impl QuantizedKvCache {
    /// An empty cache at `bits` (2, 4 or 8) with `group_size` columns per
    /// scale, growing [`DEFAULT_STEP`] positions at a time.
    pub fn new(bits: u32, group_size: usize) -> Result<Self, QuantError> {
        // Codes that divide a word never straddle two, which is what lets
        // the encoder pack with shifts and ors alone.
        if !matches!(bits, 2 | 4 | 8) {
            return Err(QuantError::UnsupportedBits(bits));
        }
        Ok(Self {
            format: QuantFormat::new(bits, group_size, QuantMode::Affine)?,
            keys: None,
            values: None,
            offset: 0,
            step: DEFAULT_STEP,
        })
    }

    /// Sets how many positions each growth adds.
    pub fn with_step(mut self, step: usize) -> Self {
        assert!(step > 0, "the growth step must be positive");
        self.step = step;
        self
    }

    /// The storage format.
    pub fn format(&self) -> QuantFormat {
        self.format
    }

    /// Positions currently held.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Bytes allocated for keys and values (capacity, not just the held
    /// positions), for budgeting a prompt cache.
    pub fn nbytes(&self) -> usize {
        self.keys.as_ref().map_or(0, PackedRows::bytes) + self.values.as_ref().map_or(0, PackedRows::bytes)
    }

    /// Appends `keys`/`values` (`[b, h, n, d]`) and returns every position
    /// held so far, decoded to `[b, h, offset + n, d]` in the input dtypes.
    ///
    /// # Panics
    /// If a head dimension is not a multiple of 32 and of the group size.
    pub fn update(&mut self, keys: Tensor<4>, values: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let (k_dtype, v_dtype) = (keys.dtype(), values.dtype());
        self.append(keys, values);
        let (k, v) = (self.keys.as_ref().expect("appended"), self.values.as_ref().expect("appended"));
        (self.decode(k, self.offset).cast(k_dtype), self.decode(v, self.offset).cast(v_dtype))
    }

    /// Appends `keys`/`values` (`[b, kv_heads, n, d]`) and returns the
    /// attention of `queries` (`[b, heads, n, d]`) over every held position,
    /// computed against the packed cache rather than a decoded copy of it.
    ///
    /// The same contract as [`cache_attention`](crate::cache_attention) over
    /// [`update`](Self::update)'s output: `heads` is a multiple of
    /// `kv_heads` and query head `j` reads kv head `j / (heads / kv_heads)`;
    /// the queries are the last `n` positions and each sees the positions up
    /// to its own; `scale` defaults to `1 / sqrt(d)`.
    ///
    /// Both products go through [`LlmOps::batched_quantized_matmul`]. On
    /// CubeCL runtimes a decode step (up to 8 query rows per kv head) reads
    /// the packed rows in place, so the full-precision cache never exists;
    /// a larger block (prefill) decodes the held rows once for a dense matmul.
    ///
    /// # Panics
    /// If `heads` is not a multiple of `kv_heads`, or a head dimension is not
    /// a multiple of 32 and of the group size.
    pub fn attend(&mut self, queries: Tensor<4>, keys: Tensor<4>, values: Tensor<4>, scale: Option<f64>) -> Tensor<4> {
        let [b, heads, n, d] = queries.dims();
        let kv_heads = keys.dims()[1];
        assert!(heads % kv_heads == 0, "{heads} query heads cannot share {kv_heads} key/value heads");
        let rep = heads / kv_heads;
        let (dtype, device) = (queries.dtype(), queries.device());
        self.append(keys, values);
        let len = self.offset;
        let scale = scale.unwrap_or(1.0 / (d as f64).sqrt());

        // Query heads grouped by the kv head they read: `[b·kv_heads, rep·n, d]`.
        let q = queries.cast(DType::F32).reshape([b * kv_heads, rep * n, d]);
        let k = self.keys.as_ref().expect("appended");
        let scores = self.packed_matmul(q, k, len, true).mul_scalar(scale);
        let mut scores = scores.reshape([b * kv_heads, rep, n, len]);
        if n > 1 {
            let mask = causal_mask(n, len, &device).expand([b * kv_heads, rep, n, len]);
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        let probs = softmax(scores, 3).reshape([b * kv_heads, rep * n, len]);
        let v = self.values.as_ref().expect("appended");
        let out = self.packed_matmul(probs, v, len, false);
        out.reshape([b, heads, n, d]).cast(dtype)
    }

    /// `x[b·h, m, ·]` against the first `len` positions of one side, read
    /// packed and in place: `x · Kᵀ` when `transpose`, else `x · V`.
    fn packed_matmul(&self, x: Tensor<3>, rows: &PackedRows, len: usize, transpose: bool) -> Tensor<3> {
        let [b, h, capacity, words] = rows.data.dims();
        let groups = rows.scales.dims()[3];
        Tensor::from_dispatch(<Dispatch as LlmOps>::batched_quantized_matmul(
            x.into_dispatch(),
            rows.data.clone().reshape([b * h, capacity, words]).into_dispatch(),
            rows.scales.clone().reshape([b * h, capacity, groups]).into_dispatch(),
            rows.biases.clone().reshape([b * h, capacity, groups]).into_dispatch(),
            len,
            self.format.bits(),
            self.format.group_size(),
            true,
            transpose,
        ))
    }

    /// Quantizes `keys`/`values` (`[b, h, n, d]`) into the next `n`
    /// positions, growing the buffers by whole steps when full.
    fn append(&mut self, keys: Tensor<4>, values: Tensor<4>) {
        let new_k = self.encode(keys);
        let new_v = self.encode(values);
        let n = new_k.data.dims()[2];
        let end = self.offset + n;

        let capacity = self.keys.as_ref().map_or(0, PackedRows::capacity);
        if end > capacity {
            let grow = end.div_ceil(self.step) * self.step - capacity;
            let k_more = empty_like(&new_k, grow);
            let v_more = empty_like(&new_v, grow);
            self.keys = Some(match self.keys.take() {
                Some(k) => k.cat(k_more),
                None => k_more,
            });
            self.values = Some(match self.values.take() {
                Some(v) => v.cat(v_more),
                None => v_more,
            });
        }
        let k = self.keys.take().expect("allocated above").write(self.offset, new_k);
        let v = self.values.take().expect("allocated above").write(self.offset, new_v);
        self.offset = end;
        self.keys = Some(k);
        self.values = Some(v);
    }

    /// Drops the last `n` positions. Returns how many were removed.
    pub fn trim(&mut self, n: usize) -> usize {
        let n = n.min(self.offset);
        self.offset -= n;
        n
    }

    /// Forgets everything; the allocation is kept for reuse.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Quantizes `[b, h, n, d]` into packed rows.
    fn encode(&self, x: Tensor<4>) -> PackedRows {
        let [b, h, n, d] = x.dims();
        self.format
            .check_columns(d)
            .unwrap_or_else(|e| panic!("head dimension {d} cannot be cached in {:?}: {e:?}", self.format));
        let bits = self.format.bits() as usize;
        let g = self.format.group_size();
        let groups = d / g;
        let per_word = 32 / bits;
        let words = d / per_word;
        let max_code = self.format.max_code() as f32;
        let dtype = x.dtype();

        let xf = x.cast(DType::F32).reshape([b, h, n, groups, g]);
        let lo = xf.clone().min_dim(4);
        let hi = xf.clone().max_dim(4);
        let scale = (hi - lo.clone()).div_scalar(max_code);
        // A constant group has scale 0: every code is 0 and decodes to `lo`.
        let safe = scale.clone().mask_fill(scale.clone().equal_elem(0.0), 1.0);
        let codes = ((xf - lo.clone()) / safe)
            .round()
            .clamp(0.0, max_code)
            .int()
            .cast(IntDType::U32)
            .reshape([b, h, n, words, per_word]);

        // Column `j` sits at bit `(j % per_word) * bits` of word `j / per_word`,
        // the layout `quant::code_at` and the decoders read.
        let mut packed = codes.clone().narrow(4, 0, 1);
        for i in 1..per_word {
            let field = codes.clone().narrow(4, i, 1).bitwise_left_shift_scalar((i * bits) as u32);
            packed = packed.bitwise_or(field);
        }

        PackedRows {
            data: packed.reshape([b, h, n, words]),
            scales: scale.reshape([b, h, n, groups]).cast(dtype),
            biases: lo.reshape([b, h, n, groups]).cast(dtype),
        }
    }

    /// The first `len` positions of one side, decoded to `[b, h, len, d]`.
    fn decode(&self, rows: &PackedRows, len: usize) -> Tensor<4> {
        let [b, h, _, words] = rows.data.dims();
        let groups = rows.scales.dims()[3];
        let d = words * 32 / self.format.bits() as usize;
        let flat = b * h * len;
        let data = rows.data.clone().narrow(2, 0, len).reshape([flat, words]);
        let scales = rows.scales.clone().narrow(2, 0, len).reshape([flat, groups]);
        let biases = rows.biases.clone().narrow(2, 0, len).reshape([flat, groups]);
        let w: Tensor<2> = Tensor::from_dispatch(<Dispatch as LlmOps>::dequantize_packed(
            data.into_dispatch(),
            scales.into_dispatch(),
            biases.into_dispatch(),
            self.format.bits(),
            self.format.group_size(),
            true,
        ));
        w.reshape([b, h, len, d])
    }
}

/// Zeroed buffers shaped like `rows` with `n` positions.
fn empty_like(rows: &PackedRows, n: usize) -> PackedRows {
    let [b, h, _, w] = rows.data.dims();
    let g = rows.scales.dims()[3];
    let device = rows.data.device();
    PackedRows {
        data: Tensor::<4, Int>::zeros([b, h, n, w], (&device, DType::U32)),
        scales: Tensor::<4>::zeros([b, h, n, g], (&device, rows.scales.dtype())),
        biases: Tensor::<4>::zeros([b, h, n, g], (&device, rows.biases.dtype())),
    }
}

/// One cache per layer.
impl PrefixState for Vec<QuantizedKvCache> {
    fn trim_to(&mut self, len: usize) -> bool {
        for cache in self.iter_mut() {
            let held = cache.offset();
            if held < len {
                return false;
            }
            cache.trim(held - len);
        }
        true
    }
}
