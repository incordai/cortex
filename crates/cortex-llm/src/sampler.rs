//! Token sampling: penalties, temperature, top-k, top-p, min-p, XTC.
//!
//! The heavy work stays on the device: penalties touch only the tokens in the
//! recent history, and the device computes the exact log-normaliser over the
//! whole vocabulary plus the top `CANDIDATES` tokens. Only those candidates
//! (a few KB) come back to the host, where the truncations are applied
//! exactly, using the true normaliser. When a cutoff would depend on tokens
//! beyond the candidates, the full distribution is read instead, so the
//! result is never an approximation.
//!
//! Callers: `generate.rs` (one `sample` per decoded token) and its
//! speculative path (`distribution` + `Distribution::prob`).

use alloc::vec;
use alloc::vec::Vec;
use cortex_core::tensor::ops::IndexingUpdateOp;
use cortex_core::tensor::{Int, Tensor, TensorData};

/// Candidates the device ranks and reads back per step.
///
/// Small and fixed on purpose: the CubeCL top-k reduction unrolls `k` into
/// the kernel (one register slot per candidate, `cubek-reduce`
/// `instructions/topk.rs`), so a large `k` builds a kernel that effectively
/// never finishes compiling, and a `k` taken from the request would compile
/// one kernel per distinct value. Anything wider — a larger `top_k`, or a
/// cutoff that reaches past these — is ranked on the host from a full read.
pub const CANDIDATES: usize = 64;

/// Sampling settings. Defaults are greedy decoding with no penalties.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerConfig {
    /// `0` is greedy (argmax).
    pub temperature: f32,
    /// Keep the `top_k` most likely tokens; `0` disables.
    pub top_k: usize,
    /// Keep the smallest set whose probability reaches `top_p`; `1` disables.
    pub top_p: f32,
    /// Drop tokens below `min_p` times the most likely one; `0` disables.
    pub min_p: f32,
    /// XTC: with this probability, remove every token at or above
    /// `xtc_threshold` except the least likely of them; `0` disables.
    pub xtc_probability: f32,
    /// See `xtc_probability`.
    pub xtc_threshold: f32,
    /// Divide positive / multiply negative logits of recent tokens; `1` disables.
    pub repetition_penalty: f32,
    /// Subtracted once from every recent token's logit; `0` disables.
    pub presence_penalty: f32,
    /// Subtracted per occurrence of a recent token; `0` disables.
    pub frequency_penalty: f32,
    /// How many recent tokens the penalties look at.
    pub penalty_window: usize,
    /// Random seed; equal seeds give equal samples.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            penalty_window: 64,
            seed: 0,
        }
    }
}

/// A small, fast, seedable generator (PCG-XSH-RR 64/32).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    const MUL: u64 = 6364136223846793005;
    const INC: u64 = 1442695040888963407;

    /// A generator for `seed`.
    pub fn new(seed: u64) -> Self {
        let mut rng = Self { state: 0 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MUL).wrapping_add(Self::INC);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        let hi = (self.next_u32() as u64) << 21;
        let lo = (self.next_u32() >> 11) as u64;
        (hi | lo) as f64 / (1u64 << 53) as f64
    }
}

/// A sparse token distribution: candidate tokens and their probabilities,
/// summing to one.
#[derive(Clone, Debug, PartialEq)]
pub struct Distribution {
    tokens: Vec<u32>,
    probs: Vec<f64>,
}

impl Distribution {
    /// A distribution proportional to non-negative `weights`.
    ///
    /// # Panics
    /// If the lengths differ or the weights do not sum to a positive value.
    pub fn from_weights(tokens: Vec<u32>, weights: Vec<f64>) -> Self {
        assert_eq!(tokens.len(), weights.len(), "one weight per token");
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "weights must sum to a positive value");
        Self {
            tokens,
            probs: weights.iter().map(|w| w / total).collect(),
        }
    }

    /// A distribution concentrated on one token.
    pub fn certain(token: u32) -> Self {
        Self {
            tokens: vec![token],
            probs: vec![1.0],
        }
    }

    /// The candidates and their probabilities.
    pub fn entries(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.tokens.iter().copied().zip(self.probs.iter().copied())
    }

    /// Probability of `token` (zero if it was truncated away).
    pub fn prob(&self, token: u32) -> f64 {
        self.tokens
            .iter()
            .position(|t| *t == token)
            .map_or(0.0, |i| self.probs[i])
    }

    /// The most likely token.
    pub fn argmax(&self) -> u32 {
        let mut best = 0;
        for i in 1..self.probs.len() {
            if self.probs[i] > self.probs[best] {
                best = i;
            }
        }
        self.tokens[best]
    }

    /// Draws a token.
    pub fn draw(&self, rng: &mut Rng) -> u32 {
        let u = rng.next_f64();
        let mut acc = 0.0;
        for (t, p) in self.entries() {
            acc += p;
            if u < acc {
                return t;
            }
        }
        *self.tokens.last().expect("a distribution has at least one token")
    }
}

/// Stateful sampler: holds the settings, the random generator and the recent
/// token history the penalties read.
#[derive(Clone, Debug)]
pub struct Sampler {
    config: SamplerConfig,
    rng: Rng,
    history: Vec<u32>,
}

impl Sampler {
    /// A sampler with these settings.
    pub fn new(config: SamplerConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self {
            config,
            rng,
            history: Vec::new(),
        }
    }

    /// The settings.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Records a token the penalties should account for (prompt or output).
    pub fn observe(&mut self, token: u32) {
        self.history.push(token);
        let window = self.config.penalty_window.max(1) * 4;
        if self.history.len() > window {
            let drop = self.history.len() - self.config.penalty_window.max(1);
            self.history.drain(..drop);
        }
    }

    /// Forgets the history (new conversation).
    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    /// Draws from `dist` with this sampler's generator (no history change).
    pub fn draw(&mut self, dist: &Distribution) -> u32 {
        dist.draw(&mut self.rng)
    }

    /// A uniform draw in `[0, 1)` from this sampler's generator.
    pub fn uniform(&mut self) -> f64 {
        self.rng.next_f64()
    }

    /// Samples the next token from `logits[vocab]` and records it.
    pub fn sample(&mut self, logits: Tensor<1>) -> u32 {
        let token = if self.config.temperature <= 0.0 {
            let logits = self.penalize(logits);
            read_ints(logits.argmax(0))[0] as u32
        } else {
            let dist = self.distribution(logits);
            dist.draw(&mut self.rng)
        };
        self.observe(token);
        token
    }

    /// The processed distribution for `logits[vocab]` (penalties,
    /// temperature and every truncation applied), without drawing from it.
    ///
    /// Greedy settings give a distribution concentrated on the argmax.
    pub fn distribution(&mut self, logits: Tensor<1>) -> Distribution {
        let logits = self.penalize(logits);
        if self.config.temperature <= 0.0 {
            return Distribution::certain(read_ints(logits.argmax(0))[0] as u32);
        }
        let vocab = logits.dims()[0];
        let scaled = logits.div_scalar(self.config.temperature);
        let max = scaled.clone().max();
        let lse = (scaled.clone() - max.clone()).exp().sum().log() + max;
        let top_k = self.config.top_k;
        let k = CANDIDATES.min(vocab);
        let (values, indices) = scaled.clone().topk_with_indices(k, 0);
        let lse = read_floats(lse)[0] as f64;
        let values = read_floats(values);
        let indices = read_ints(indices);

        let mut cand: Vec<(u32, f64)> = indices
            .iter()
            .zip(&values)
            .map(|(i, v)| (*i as u32, (*v as f64 - lse).exp()))
            .collect();

        // The candidates decide everything when `top_k` keeps no more than
        // them, or when the remaining cutoffs land inside them.
        let wider_top_k = top_k == 0 || top_k > k;
        if k < vocab && wider_top_k && !self.candidates_suffice(&cand) {
            let all = read_floats(scaled);
            cand = all
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, (*v as f64 - lse).exp()))
                .collect();
            if top_k > 0 && top_k < cand.len() {
                cand.select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
                cand.truncate(top_k);
            }
            if self.needs_order() {
                cand.sort_by(|a, b| b.1.total_cmp(&a.1));
            }
        }
        if top_k > 0 {
            cand.truncate(top_k);
        }
        self.truncate(cand)
    }

    /// Whether a truncation reads the candidates in descending order (a plain
    /// draw does not, so a full-vocabulary read skips the sort).
    fn needs_order(&self) -> bool {
        let c = &self.config;
        c.top_k > 0 || c.top_p < 1.0 || c.min_p > 0.0 || c.xtc_probability > 0.0
    }

    /// Whether the top candidates alone decide every cutoff exactly: either
    /// they already hold (nearly) all the probability, or top-p / min-p cut
    /// inside them, so no tail token could survive.
    fn candidates_suffice(&self, cand: &[(u32, f64)]) -> bool {
        let covered: f64 = cand.iter().map(|c| c.1).sum();
        if covered >= 1.0 - 1e-6 {
            return true;
        }
        let smallest = cand.last().map_or(0.0, |c| c.1);
        let max = cand.first().map_or(0.0, |c| c.1);
        let top_p_inside = self.config.top_p < 1.0 && covered >= self.config.top_p as f64;
        let min_p_inside = self.config.min_p > 0.0 && smallest < self.config.min_p as f64 * max;
        top_p_inside || min_p_inside
    }

    /// Applies min-p, top-p and XTC to candidates sorted by descending
    /// probability, then renormalizes.
    fn truncate(&mut self, mut cand: Vec<(u32, f64)>) -> Distribution {
        if self.config.min_p > 0.0 {
            let floor = self.config.min_p as f64 * cand[0].1;
            cand.retain(|c| c.1 >= floor);
        }
        if self.config.top_p < 1.0 {
            let total: f64 = cand.iter().map(|c| c.1).sum();
            let target = self.config.top_p as f64 * total;
            let mut acc = 0.0;
            let mut keep = cand.len();
            for (i, c) in cand.iter().enumerate() {
                acc += c.1;
                if acc >= target {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
        }
        if self.config.xtc_probability > 0.0
            && self.rng.next_f64() < self.config.xtc_probability as f64
        {
            let total: f64 = cand.iter().map(|c| c.1).sum();
            let above = cand
                .iter()
                .take_while(|c| c.1 / total >= self.config.xtc_threshold as f64)
                .count();
            if above > 1 {
                cand.drain(..above - 1);
            }
        }
        let total: f64 = cand.iter().map(|c| c.1).sum();
        Distribution {
            tokens: cand.iter().map(|c| c.0).collect(),
            probs: cand.iter().map(|c| c.1 / total).collect(),
        }
    }

    /// Applies the repetition, presence and frequency penalties to the
    /// tokens of the recent history, on the device.
    fn penalize(&self, logits: Tensor<1>) -> Tensor<1> {
        let c = &self.config;
        if self.history.is_empty()
            || (c.repetition_penalty == 1.0 && c.presence_penalty == 0.0 && c.frequency_penalty == 0.0)
        {
            return logits;
        }
        let start = self.history.len().saturating_sub(c.penalty_window.max(1));
        let mut counts: Vec<(u32, u32)> = Vec::new();
        for &t in &self.history[start..] {
            match counts.iter_mut().find(|e| e.0 == t) {
                Some(e) => e.1 += 1,
                None => counts.push((t, 1)),
            }
        }
        let device = logits.device();
        let ids: Vec<i64> = counts.iter().map(|e| e.0 as i64).collect();
        let n = ids.len();
        let ids = Tensor::<1, Int>::from_data(TensorData::new(ids, [n]), &device);
        let freq: Vec<f32> = counts.iter().map(|e| e.1 as f32).collect();
        let freq = Tensor::<1>::from_data(TensorData::new(freq, [n]), &device);

        let picked = logits.clone().select(0, ids.clone());
        let positive = picked.clone().greater_equal_elem(0.0);
        let rep = picked
            .clone()
            .div_scalar(c.repetition_penalty)
            .mask_where(positive.bool_not(), picked.mul_scalar(c.repetition_penalty));
        let adjusted = rep.sub_scalar(c.presence_penalty) - freq.mul_scalar(c.frequency_penalty);
        logits.select_assign(0, ids, adjusted, IndexingUpdateOp::Assign)
    }
}

fn read_floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data()
        .try_to_vec::<f32>()
        .expect("sampler reads f32 logits")
}

fn read_ints<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.into_data()
        .convert::<i64>()
        .try_to_vec::<i64>()
        .expect("sampler reads integer indices")
}
