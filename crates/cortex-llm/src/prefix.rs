//! Prompt (prefix) cache: reuse the model state of a previous request whose
//! tokens share a prefix with the new one, so only the new suffix is
//! prefilled.
//!
//! Entries are evicted least-recently-used under a byte budget. A request can
//! reuse an entry that is longer than their common prefix when the state can
//! be trimmed back ([`PrefixState::trim_to`]), which is what lets two
//! conversations that diverge after a shared system prompt share it.
//!
//! Callers: the serving layer, once per request. Calls: `cache.rs`
//! (`KvCache::trim` through the `Vec<KvCache>` impl).

use alloc::vec::Vec;

use crate::KvCache;

/// A model state that can be cut back to its first `len` positions.
pub trait PrefixState: Clone {
    /// Keeps the first `len` positions. Returns `false` when the state
    /// cannot be trimmed (it is then not reused).
    fn trim_to(&mut self, len: usize) -> bool;
}

/// One cache per layer.
impl PrefixState for Vec<KvCache> {
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

#[derive(Clone, Debug)]
struct Entry<S> {
    tokens: Vec<u32>,
    state: S,
    bytes: usize,
    last_used: u64,
}

/// An LRU cache of model states keyed by the token sequence that produced
/// them.
#[derive(Clone, Debug)]
pub struct PromptCache<S> {
    entries: Vec<Entry<S>>,
    capacity_bytes: usize,
    used_bytes: usize,
    clock: u64,
}

/// The state to resume from, and how many prompt tokens it already covers.
#[derive(Clone, Debug)]
pub struct PrefixHit<S> {
    /// Model state covering the first `reused` tokens of the prompt.
    pub state: S,
    /// Prompt tokens already covered; prefill only the rest.
    pub reused: usize,
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl<S: PrefixState> PromptCache<S> {
    /// An empty cache holding at most `capacity_bytes` of state.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity_bytes,
            used_bytes: 0,
            clock: 0,
        }
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes of state held.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Stores the state reached after `tokens`, replacing an entry for the
    /// same tokens and evicting least-recently-used entries to fit. A state
    /// larger than the whole budget is not stored.
    pub fn insert(&mut self, tokens: Vec<u32>, state: S, bytes: usize) {
        if bytes > self.capacity_bytes {
            return;
        }
        if let Some(i) = self.entries.iter().position(|e| e.tokens == tokens) {
            let old = self.entries.swap_remove(i);
            self.used_bytes -= old.bytes;
        }
        while self.used_bytes + bytes > self.capacity_bytes {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .expect("over budget implies at least one entry");
            let evicted = self.entries.swap_remove(oldest);
            self.used_bytes -= evicted.bytes;
        }
        self.clock += 1;
        self.used_bytes += bytes;
        self.entries.push(Entry {
            tokens,
            state,
            bytes,
            last_used: self.clock,
        });
    }

    /// The state sharing the longest prefix with `prompt`, trimmed to that
    /// prefix. At least one prompt token is always left to prefill, so the
    /// caller has logits to sample from.
    ///
    /// Ties go to the shortest entry: it needs the least trimming, and it is
    /// the one whose recency should be refreshed — refreshing a longer entry
    /// that merely shares the prefix would let the exact match go stale and be
    /// evicted first.
    pub fn lookup(&mut self, prompt: &[u32]) -> Option<PrefixHit<S>> {
        let limit = prompt.len().saturating_sub(1);
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let shared = common_prefix(&e.tokens, prompt).min(limit);
            if shared == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some((b, b_shared)) => {
                    shared > b_shared
                        || (shared == b_shared && e.tokens.len() < self.entries[b].tokens.len())
                }
            };
            if better {
                best = Some((i, shared));
            }
        }
        let (i, shared) = best?;
        self.clock += 1;
        self.entries[i].last_used = self.clock;
        let mut state = self.entries[i].state.clone();
        if self.entries[i].tokens.len() > shared && !state.trim_to(shared) {
            return None;
        }
        Some(PrefixHit {
            state,
            reused: shared,
        })
    }
}
