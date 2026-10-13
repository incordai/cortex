//! Generation, speculative decoding and the prompt cache, with toy bigram
//! models whose logits depend only on the previous token.

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::{
    GenerateOptions, LanguageModel, PrefixState, PromptCache, Sampler, SamplerConfig, StopReason,
    generate, speculative_generate,
};

const VOCAB: usize = 8;

/// `logits(next | prev) = table[prev]`. Records every token it was fed so
/// tests can check what the model saw, and honours `trim`.
struct Bigram {
    table: Vec<f32>,
    fed: Vec<u32>,
    forwards: usize,
}

impl Bigram {
    fn new(seed: u64) -> Self {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let table = (0..VOCAB * VOCAB)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 4.0
            })
            .collect();
        Self { table, fed: Vec::new(), forwards: 0 }
    }
}

impl LanguageModel for Bigram {
    fn forward(&mut self, tokens: &[u32]) -> Tensor<2> {
        self.forwards += 1;
        self.fed.extend_from_slice(tokens);
        let rows: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| self.table[t as usize * VOCAB..(t as usize + 1) * VOCAB].to_vec())
            .collect();
        Tensor::from_data(TensorData::new(rows, [tokens.len(), VOCAB]), &Device::default())
    }

    fn trim(&mut self, n: usize) {
        let keep = self.fed.len() - n;
        self.fed.truncate(keep);
    }
}

fn greedy_by_hand(model: &Bigram, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut prev = *prompt.last().unwrap();
    (0..n)
        .map(|_| {
            let row = &model.table[prev as usize * VOCAB..(prev as usize + 1) * VOCAB];
            let best = (0..VOCAB).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() as u32;
            prev = best;
            best
        })
        .collect()
}

#[test]
fn greedy_generation_follows_the_argmax_chain() {
    let mut model = Bigram::new(1);
    let reference = greedy_by_hand(&model, &[3], 12);
    let mut sampler = Sampler::new(SamplerConfig::default());
    let opts = GenerateOptions { max_tokens: 12, stop_tokens: vec![] };
    let (out, reason) = generate(&mut model, &[3], &mut sampler, &opts, |_| true);
    assert_eq!(out, reference);
    assert_eq!(reason, StopReason::MaxTokens);
    // Prompt, then every generated token but the last was fed back.
    assert_eq!(model.fed.len(), 1 + 11);
}

#[test]
fn stop_tokens_and_cancellation_end_generation() {
    let model_ref = Bigram::new(1);
    let chain = greedy_by_hand(&model_ref, &[3], 12);
    let stop = chain[4];
    let first_stop = chain.iter().position(|&t| t == stop).unwrap();

    let mut model = Bigram::new(1);
    let mut sampler = Sampler::new(SamplerConfig::default());
    let opts = GenerateOptions { max_tokens: 12, stop_tokens: vec![stop] };
    let (out, reason) = generate(&mut model, &[3], &mut sampler, &opts, |_| true);
    assert_eq!(out, chain[..=first_stop]);
    assert_eq!(reason, StopReason::StopToken(stop));

    let mut model = Bigram::new(1);
    let mut sampler = Sampler::new(SamplerConfig::default());
    let opts = GenerateOptions { max_tokens: 12, stop_tokens: vec![] };
    let mut seen = 0;
    let (out, reason) = generate(&mut model, &[3], &mut sampler, &opts, |_| {
        seen += 1;
        seen < 3
    });
    assert_eq!(out.len(), 3);
    assert_eq!(reason, StopReason::Cancelled);
}

#[test]
fn speculative_greedy_reproduces_the_target_exactly() {
    for draft_seed in [2u64, 7, 1] {
        let mut target = Bigram::new(1);
        let mut draft = Bigram::new(draft_seed);
        let reference = greedy_by_hand(&target, &[5], 20);
        let mut sampler = Sampler::new(SamplerConfig::default());
        let opts = GenerateOptions { max_tokens: 20, stop_tokens: vec![] };
        let (out, _) = speculative_generate(&mut target, &mut draft, &[5], &mut sampler, &opts, 4, |_| true);
        assert_eq!(out, reference, "draft seed {draft_seed}");
        // Both models end holding the prompt plus every accepted token but
        // the final one, never a rejected proposal.
        assert_eq!(target.fed, [&[5u32][..], &out[..out.len() - 1]].concat());
        assert_eq!(draft.fed, target.fed);
    }
}

#[test]
fn an_identical_draft_is_always_accepted() {
    let mut target = Bigram::new(3);
    let mut draft = Bigram::new(3);
    let mut sampler = Sampler::new(SamplerConfig::default());
    let opts = GenerateOptions { max_tokens: 20, stop_tokens: vec![] };
    let (out, _) = speculative_generate(&mut target, &mut draft, &[1], &mut sampler, &opts, 4, |_| true);
    assert_eq!(out.len(), 20);
    // 20 tokens at 5 per round (4 accepted proposals + 1 bonus): four
    // verification forwards after the prompt, one prompt forward, and one
    // forward of each round's bonus token except the last.
    assert_eq!(target.forwards, 1 + 4 + 3);
}

/// A state that is just the token count, trimmable to any shorter length.
#[derive(Clone, Debug, PartialEq)]
struct Len(usize);

impl PrefixState for Len {
    fn trim_to(&mut self, len: usize) -> bool {
        if len > self.0 {
            return false;
        }
        self.0 = len;
        true
    }
}

#[test]
fn prompt_cache_reuses_the_longest_prefix_and_evicts_lru() {
    let mut cache = PromptCache::new(100);
    cache.insert(vec![1, 2, 3, 4], Len(4), 40);
    cache.insert(vec![1, 2, 9], Len(3), 40);

    // Shares 3 tokens with the first entry: trimmed back to 3.
    let hit = cache.lookup(&[1, 2, 3, 7, 7]).unwrap();
    assert_eq!((hit.reused, hit.state), (3, Len(3)));

    // A prompt equal to an entry still leaves one token to prefill.
    let hit = cache.lookup(&[1, 2, 9]).unwrap();
    assert_eq!(hit.reused, 2);

    assert!(cache.lookup(&[5, 5]).is_none());

    // Over budget: the least recently used ([1,2,3,4] was used before
    // [1,2,9]) is evicted.
    cache.insert(vec![8, 8], Len(2), 40);
    assert_eq!(cache.len(), 2);
    assert!(cache.lookup(&[1, 2, 3, 4, 5]).unwrap().reused == 2, "only [1,2,9]'s prefix remains");
    assert!(cache.used_bytes() <= 100);
}
