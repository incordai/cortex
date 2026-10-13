//! Autoregressive generation and speculative decoding over any model that
//! implements [`LanguageModel`].
//!
//! Callers: applications and the serving layer. Calls: the model's
//! `forward`/`trim`, and `sampler.rs` for every token choice.

use alloc::vec::Vec;
use cortex_core::tensor::Tensor;

use crate::sampler::{Distribution, Sampler};

/// A causal language model that owns its caches.
pub trait LanguageModel {
    /// Feeds `tokens` after everything already cached and returns the logits
    /// predicting the token after each of them, `[tokens.len(), vocab]`.
    fn forward(&mut self, tokens: &[u32]) -> Tensor<2>;

    /// Drops the last `n` cached positions (rejected speculative tokens).
    fn trim(&mut self, n: usize);
}

/// Limits for one generation.
#[derive(Clone, Debug, Default)]
pub struct GenerateOptions {
    /// Stop after this many new tokens.
    pub max_tokens: usize,
    /// Stop when one of these is produced (it is included in the output).
    pub stop_tokens: Vec<u32>,
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// `max_tokens` new tokens were produced.
    MaxTokens,
    /// A stop token was produced.
    StopToken(u32),
    /// The token callback asked to stop.
    Cancelled,
}

fn row(logits: &Tensor<2>, i: usize) -> Tensor<1> {
    let vocab = logits.dims()[1];
    logits.clone().narrow(0, i, 1).reshape([vocab])
}

fn last_row(logits: &Tensor<2>) -> Tensor<1> {
    row(logits, logits.dims()[0] - 1)
}

/// Generates up to `options.max_tokens` tokens after `prompt`.
///
/// `on_token` sees every new token as it is produced and returns `false` to
/// cancel. The prompt is recorded in the sampler's penalty history.
pub fn generate<M: LanguageModel>(
    model: &mut M,
    prompt: &[u32],
    sampler: &mut Sampler,
    options: &GenerateOptions,
    mut on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, StopReason) {
    assert!(!prompt.is_empty(), "generation needs at least one prompt token");
    for &t in prompt {
        sampler.observe(t);
    }
    let mut out = Vec::new();
    let mut logits = last_row(&model.forward(prompt));
    while out.len() < options.max_tokens {
        let token = sampler.sample(logits);
        out.push(token);
        if !on_token(token) {
            return (out, StopReason::Cancelled);
        }
        if options.stop_tokens.contains(&token) {
            return (out, StopReason::StopToken(token));
        }
        if out.len() == options.max_tokens {
            break;
        }
        logits = last_row(&model.forward(&[token]));
    }
    (out, StopReason::MaxTokens)
}

/// `max(0, p - q)`, renormalized; `p` itself when the two coincide.
fn residual(p: &Distribution, q: &Distribution) -> Distribution {
    let mut tokens = Vec::new();
    let mut weights = Vec::new();
    for (t, pt) in p.entries() {
        let w = pt - q.prob(t);
        if w > 0.0 {
            tokens.push(t);
            weights.push(w);
        }
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return p.clone();
    }
    Distribution::from_weights(tokens, weights)
}

/// Speculative decoding: `draft` proposes `num_draft` tokens, `target`
/// verifies them in one batched forward, and each proposal is accepted with
/// probability `min(1, p/q)`; the first rejection is replaced by a sample of
/// `max(0, p - q)`. The output has exactly the distribution of sampling
/// `target` alone (Leviathan et al.); with greedy settings it is exactly
/// target's greedy output.
///
/// Both models must share a vocabulary. Penalties stay exact: each
/// distribution is computed with the history of every token before it,
/// drafted ones included, and only accepted tokens are committed.
#[allow(clippy::too_many_arguments)]
pub fn speculative_generate<T: LanguageModel, D: LanguageModel>(
    target: &mut T,
    draft: &mut D,
    prompt: &[u32],
    sampler: &mut Sampler,
    options: &GenerateOptions,
    num_draft: usize,
    mut on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, StopReason) {
    assert!(!prompt.is_empty(), "generation needs at least one prompt token");
    assert!(num_draft > 0, "speculation needs at least one draft token");
    for &t in prompt {
        sampler.observe(t);
    }
    let mut out = Vec::new();
    let mut target_next = last_row(&target.forward(prompt));
    let mut draft_next = last_row(&draft.forward(prompt));

    while out.len() < options.max_tokens {
        let k = num_draft.min(options.max_tokens - out.len());

        // Draft k proposals, each from the history including the previous ones.
        let mut drafting = sampler.clone();
        let mut proposals = Vec::with_capacity(k);
        let mut q = Vec::with_capacity(k);
        let mut logits = Some(draft_next.clone());
        for i in 0..k {
            let dist = drafting.distribution(logits.take().expect("set for every drafted position"));
            let x = sampler.draw(&dist);
            drafting.observe(x);
            proposals.push(x);
            q.push(dist);
            if i + 1 < k {
                logits = Some(last_row(&draft.forward(&[x])));
            }
        }
        // The draft has consumed proposals[..k-1]; feed the last so both
        // models hold the same positions before verification.
        draft.forward(&proposals[k - 1..]);

        // Target scores every proposal in one forward.
        let verify = target.forward(&proposals);
        let mut checking = sampler.clone();
        let mut accepted = 0;
        let mut next = None;
        for i in 0..k {
            let logits_i = if i == 0 { target_next.clone() } else { row(&verify, i - 1) };
            let p = checking.distribution(logits_i);
            let x = proposals[i];
            let ratio = if q[i].prob(x) > 0.0 { p.prob(x) / q[i].prob(x) } else { 0.0 };
            if sampler.uniform() < ratio.min(1.0) {
                accepted += 1;
                checking.observe(x);
            } else {
                next = Some(sampler.draw(&residual(&p, &q[i])));
                break;
            }
        }
        let next = match next {
            Some(t) => t,
            None => {
                let p = checking.distribution(row(&verify, k - 1));
                sampler.draw(&p)
            }
        };

        // Roll both models back to the accepted prefix.
        target.trim(k - accepted);
        draft.trim(k - accepted);

        let mut stop = None;
        for &t in proposals[..accepted].iter().chain(core::iter::once(&next)) {
            sampler.observe(t);
            out.push(t);
            if !on_token(t) {
                stop = Some(StopReason::Cancelled);
            } else if options.stop_tokens.contains(&t) {
                stop = Some(StopReason::StopToken(t));
            } else if out.len() == options.max_tokens {
                stop = Some(StopReason::MaxTokens);
            }
            if stop.is_some() {
                break;
            }
        }
        if let Some(reason) = stop {
            return (out, reason);
        }
        target_next = last_row(&target.forward(&[next]));
        draft_next = last_row(&draft.forward(&[next]));
    }
    (out, StopReason::MaxTokens)
}
