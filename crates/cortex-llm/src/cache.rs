//! Key/value caches for autoregressive decoding, and attention over them.
//!
//! - [`KvCache`]: grows in `step`-token chunks and writes new keys and values
//!   in place, so a decode step costs one slice write rather than a copy of
//!   the whole history (the `Tensor::cat` per step this replaces is O(n²) over
//!   a generation).
//! - [`RotatingKvCache`]: a fixed window (sliding-window attention) that keeps
//!   the first `keep` positions (attention sinks) and overwrites the oldest of
//!   the rest.
//! - [`cache_attention`]: grouped-query attention of new queries against a
//!   cache, with the causal mask aligned to the cache offset.
//!
//! Callers: model code, once per attention layer per step; `prefix.rs` saves
//! and restores [`KvCache`] states. Sibling: `cortex-nn`'s `MhaCache` /
//! `TensorCache` (training-oriented, concat per step, returns weights).

use alloc::vec::Vec;
use cortex_core::tensor::{Bool, Tensor, TensorData, module::attention};
use cortex_core::tensor::ops::AttentionModuleOptions;

/// Default growth chunk, in tokens.
pub const DEFAULT_STEP: usize = 256;

/// A growable key/value cache for one attention layer.
///
/// Keys and values are `[batch, kv_heads, positions, head_dim]`.
#[derive(Clone, Debug)]
pub struct KvCache {
    keys: Option<Tensor<4>>,
    values: Option<Tensor<4>>,
    offset: usize,
    step: usize,
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new(DEFAULT_STEP)
    }
}

impl KvCache {
    /// An empty cache that grows `step` positions at a time.
    pub fn new(step: usize) -> Self {
        assert!(step > 0, "the growth step must be positive");
        Self {
            keys: None,
            values: None,
            offset: 0,
            step,
        }
    }

    /// Positions currently held.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Appends `keys`/`values` (`[b, h, n, d]`) and returns every position
    /// held so far, `[b, h, offset + n, d]`.
    pub fn update(&mut self, keys: Tensor<4>, values: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let [b, h, n, dk] = keys.dims();
        let dv = values.dims()[3];
        let end = self.offset + n;
        let capacity = self.keys.as_ref().map_or(0, |k| k.dims()[2]);
        if end > capacity {
            let grow = end.div_ceil(self.step) * self.step - capacity;
            let device = keys.device();
            let k_new = Tensor::<4>::zeros([b, h, grow, dk], (&device, keys.dtype()));
            let v_new = Tensor::<4>::zeros([b, h, grow, dv], (&device, values.dtype()));
            self.keys = Some(match self.keys.take() {
                Some(k) => Tensor::cat(alloc::vec![k, k_new], 2),
                None => k_new,
            });
            self.values = Some(match self.values.take() {
                Some(v) => Tensor::cat(alloc::vec![v, v_new], 2),
                None => v_new,
            });
        }
        let k = self.keys.take().expect("allocated above");
        let v = self.values.take().expect("allocated above");
        let k = k.slice_assign([0..b, 0..h, self.offset..end, 0..dk], keys);
        let v = v.slice_assign([0..b, 0..h, self.offset..end, 0..dv], values);
        self.offset = end;
        let out = (k.clone().narrow(2, 0, end), v.clone().narrow(2, 0, end));
        self.keys = Some(k);
        self.values = Some(v);
        out
    }

    /// Drops the last `n` positions (rejected speculative tokens). Returns
    /// how many were actually removed.
    pub fn trim(&mut self, n: usize) -> usize {
        let n = n.min(self.offset);
        self.offset -= n;
        n
    }

    /// Forgets everything; the allocation is kept for reuse.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// The held positions, for saving into a prefix cache.
    pub fn state(&self) -> Option<(Tensor<4>, Tensor<4>)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) if self.offset > 0 => Some((
                k.clone().narrow(2, 0, self.offset),
                v.clone().narrow(2, 0, self.offset),
            )),
            _ => None,
        }
    }

    /// A cache holding exactly `keys`/`values` (`[b, h, n, d]`).
    pub fn from_state(keys: Tensor<4>, values: Tensor<4>, step: usize) -> Self {
        let mut cache = Self::new(step);
        cache.update(keys, values);
        cache
    }
}

/// A fixed-size sliding-window cache.
///
/// Holds at most `max_size` positions: the first `keep` ever written, plus the
/// most recent `max_size - keep`. Single-token updates overwrite the oldest
/// recent slot in place; multi-token updates (prefill) append and then trim.
/// Positions come back in no particular order after the window wraps, which
/// is fine for attention because keys already carry their rotary position.
#[derive(Clone, Debug)]
pub struct RotatingKvCache {
    keys: Option<Tensor<4>>,
    values: Option<Tensor<4>>,
    keep: usize,
    max_size: usize,
    /// Total positions ever written (the next absolute position).
    offset: usize,
    /// Next ring slot to overwrite, in `keep..max_size`.
    cursor: usize,
}

impl RotatingKvCache {
    /// A window of `max_size` positions that never evicts the first `keep`.
    pub fn new(max_size: usize, keep: usize) -> Self {
        assert!(keep < max_size, "keep ({keep}) must be below the window size ({max_size})");
        Self {
            keys: None,
            values: None,
            keep,
            max_size,
            offset: 0,
            cursor: keep,
        }
    }

    /// Absolute positions written so far.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Positions currently held (at most the window size).
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.dims()[2])
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes `keys`/`values` (`[b, h, n, d]`) and returns the window to
    /// attend over.
    ///
    /// For a multi-token update the returned tensors contain every new
    /// position (possibly more than the window) so the prompt can attend to
    /// itself; the stored window is trimmed afterwards.
    pub fn update(&mut self, keys: Tensor<4>, values: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let n = keys.dims()[2];
        if n == 1 && self.len() == self.max_size {
            let [b, h, _, dk] = keys.dims();
            let dv = values.dims()[3];
            let slot = self.cursor;
            let k = self.keys.take().expect("full window").slice_assign([0..b, 0..h, slot..slot + 1, 0..dk], keys);
            let v = self.values.take().expect("full window").slice_assign([0..b, 0..h, slot..slot + 1, 0..dv], values);
            self.cursor = if slot + 1 == self.max_size { self.keep } else { slot + 1 };
            self.offset += 1;
            self.keys = Some(k.clone());
            self.values = Some(v.clone());
            return (k, v);
        }

        let all_k = match self.keys.take() {
            Some(k) => Tensor::cat(alloc::vec![self.reorder(k), keys], 2),
            None => keys,
        };
        let all_v = match self.values.take() {
            Some(v) => Tensor::cat(alloc::vec![self.reorder(v), values], 2),
            None => values,
        };
        self.offset += n;
        let total = all_k.dims()[2];
        let (k, v) = if total > self.max_size {
            let recent = self.max_size - self.keep;
            let trim = |t: Tensor<4>| {
                Tensor::cat(
                    alloc::vec![t.clone().narrow(2, 0, self.keep), t.narrow(2, total - recent, recent)],
                    2,
                )
            };
            (trim(all_k.clone()), trim(all_v.clone()))
        } else {
            (all_k.clone(), all_v.clone())
        };
        self.cursor = if k.dims()[2] == self.max_size { self.keep } else { k.dims()[2] };
        self.keys = Some(k);
        self.values = Some(v);
        (all_k, all_v)
    }

    /// Puts the held window back in chronological order (the sinks, then the
    /// ring starting at its oldest slot) before appending to it.
    fn reorder(&self, t: Tensor<4>) -> Tensor<4> {
        let len = t.dims()[2];
        if len < self.max_size || self.cursor == self.keep {
            return t;
        }
        let sinks = t.clone().narrow(2, 0, self.keep);
        let older = t.clone().narrow(2, self.cursor, len - self.cursor);
        let newer = t.narrow(2, self.keep, self.cursor - self.keep);
        Tensor::cat(alloc::vec![sinks, older, newer], 2)
    }
}

/// Attention of new queries `[b, q_heads, n, d]` against cached keys/values
/// `[b, kv_heads, len, d]`, where the queries are the last `n` of the `len`
/// positions.
///
/// Grouped-query attention: each key/value head serves
/// `q_heads / kv_heads` query heads. The causal mask is aligned to the end of
/// the cache, so query `i` sees keys `0 ..= len - n + i`; a single decode
/// query needs no mask at all.
pub fn cache_attention(
    query: Tensor<4>,
    keys: Tensor<4>,
    values: Tensor<4>,
    scale: Option<f64>,
) -> Tensor<4> {
    let [b, hq, n, _] = query.dims();
    let [_, hkv, len, dk] = keys.dims();
    let dv = values.dims()[3];
    assert!(hq % hkv == 0, "{hq} query heads cannot share {hkv} key/value heads");
    assert!(n <= len, "{n} queries but only {len} cached positions");

    let (keys, values) = if hq == hkv {
        (keys, values)
    } else {
        let rep = hq / hkv;
        let expand = |t: Tensor<4>, d: usize| {
            t.reshape([b, hkv, 1, len, d])
                .expand([b, hkv, rep, len, d])
                .reshape([b, hq, len, d])
        };
        (expand(keys, dk), expand(values, dv))
    };

    let options = AttentionModuleOptions {
        scale,
        softcap: None,
        is_causal: n > 1 && n == len,
    };
    let mask = (n > 1 && n < len).then(|| causal_mask(n, len, &query.device()).expand([b, hq, n, len]));
    attention(query, keys, values, mask, None, options)
}

/// `[1, 1, n, len]` mask, `true` where key `j` lies in the future of query
/// `i` (`j > len - n + i`). Callers: `cache_attention` above and
/// `QuantizedKvCache::attend` in `qcache.rs`.
pub(crate) fn causal_mask(n: usize, len: usize, device: &cortex_core::tensor::Device) -> Tensor<4, Bool> {
    let offset = len - n;
    let mut data = Vec::with_capacity(n * len);
    for i in 0..n {
        for j in 0..len {
            data.push(j > offset + i);
        }
    }
    Tensor::<4, Bool>::from_data(TensorData::new(data, [1, 1, n, len]), device)
}
