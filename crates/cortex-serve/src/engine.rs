//! The generation engine: one worker thread per model, running every active
//! request together (continuous batching).
//!
//! Each iteration of the worker loop
//! 1. admits queued requests up to `max_batch`, resuming each from the
//!    longest cached prefix of its prompt (`cortex_llm::PromptCache`);
//! 2. advances every request still prefilling by one chunk of at most
//!    `prefill_chunk` tokens, so a long prompt never stalls the others;
//! 3. runs one batched decode step ([`ServeModel::forward_batch`]) for every
//!    request that has a token to feed, and samples each row with that
//!    request's own sampler;
//! 4. retires finished or cancelled requests, storing their state in the
//!    prompt cache.
//!
//! Requests arrive over a channel and stream [`Event`]s back; dropping the
//! receiver cancels the request at the next iteration.
//!
//! The worker is supervised: if the model panics, the requests it was
//! running receive [`Event::Failed`], a fresh model is built with the
//! factory, and queued requests go on being served. More than
//! [`MAX_RESTARTS`] crashes within [`RESTART_WINDOW`] stop the engine
//! ([`EngineHandle::is_alive`] turns false), so a model that cannot run
//! does not crash-loop; the registry then reloads it on the next request.
//!
//! Callers: the HTTP handlers in `api/` through [`EngineHandle`]; the model
//! registry in `registry.rs` spawns one engine per loaded model. Calls:
//! [`ServeModel`], `cortex_llm::Sampler`, `cortex_llm::PromptCache`,
//! `constrain.rs` for constrained decoding.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use cortex_core::tensor::{Bool, Tensor, TensorData};
use cortex_llm::{PromptCache, Sampler, SamplerConfig};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::constrain::Constraint;
use crate::model::{ServeModel, StreamDecoder, Tokenizer};

/// Scheduling limits.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Requests generating at the same time.
    pub max_batch: usize,
    /// Prompt tokens fed per request per iteration.
    pub prefill_chunk: usize,
    /// Bytes of model state the prompt cache may keep; `0` disables it.
    pub prompt_cache_bytes: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batch: 8,
            prefill_chunk: 512,
            prompt_cache_bytes: 0,
        }
    }
}

/// One generation to run.
pub struct GenerationRequest {
    /// Prompt token ids (at least one).
    pub prompt: Vec<u32>,
    /// Sampling settings.
    pub sampling: SamplerConfig,
    /// New tokens at most.
    pub max_tokens: usize,
    /// Tokens that end the generation (not emitted as text).
    pub stop_tokens: Vec<u32>,
    /// Strings that end the generation (not emitted).
    pub stop_strings: Vec<String>,
    /// Restricts which tokens may be produced.
    pub constraint: Option<Box<dyn Constraint>>,
}

/// Why a generation ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token was produced.
    Stop,
    /// One of the stop strings appeared; it is not part of the output.
    StopString(String),
    /// `max_tokens` were produced.
    Length,
}

/// Token counts of a finished generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: usize,
    /// Of those, how many were served from the prompt cache.
    pub cached_tokens: usize,
    /// Generated tokens.
    pub completion_tokens: usize,
}

/// What a request receives, in order: any number of `Text`, then one `Done`
/// or `Error`.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Newly generated text.
    Text(String),
    /// The generation ended.
    Done {
        /// Why.
        reason: FinishReason,
        /// Token counts.
        usage: Usage,
    },
    /// The request was rejected.
    Error(String),
    /// The model failed while running the request (a server-side fault, not
    /// the request's).
    Failed(String),
}

/// Crashes tolerated within [`RESTART_WINDOW`] before the engine stops.
pub const MAX_RESTARTS: usize = 3;
/// The window [`MAX_RESTARTS`] counts crashes in.
pub const RESTART_WINDOW: Duration = Duration::from_secs(60);

struct Job {
    request: GenerationRequest,
    events: UnboundedSender<Event>,
}

/// Submits requests to an engine. Cloning is cheap; the engine stops when
/// every handle is dropped and its requests have finished.
#[derive(Clone)]
pub struct EngineHandle {
    jobs: Sender<Job>,
    alive: Arc<AtomicBool>,
}

/// The engine's worker has stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineClosed;

impl EngineHandle {
    /// Queues `request`; its events arrive on the returned receiver.
    pub fn submit(&self, request: GenerationRequest) -> Result<UnboundedReceiver<Event>, EngineClosed> {
        if !self.is_alive() {
            return Err(EngineClosed);
        }
        let (events, rx) = unbounded_channel();
        self.jobs.send(Job { request, events }).map_err(|_| EngineClosed)?;
        Ok(rx)
    }

    /// Whether the engine still serves requests (false once it stopped
    /// after repeated crashes).
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
}

/// Starts a supervised worker thread that builds its model with `factory`
/// and serves requests until every [`EngineHandle`] is dropped. `factory`
/// runs again to rebuild the model after a crash.
///
/// The model is built on the worker thread, so neither it nor its tensors
/// need to cross threads.
pub fn spawn<M, F>(factory: F, tokenizer: Arc<dyn Tokenizer>, config: EngineConfig) -> EngineHandle
where
    M: ServeModel + 'static,
    F: Fn() -> M + Send + 'static,
{
    assert!(config.max_batch > 0 && config.prefill_chunk > 0, "batch and chunk sizes must be positive");
    let (jobs, rx) = mpsc::channel();
    let alive = Arc::new(AtomicBool::new(true));
    let flag = alive.clone();
    std::thread::Builder::new()
        .name("cortex-serve-engine".into())
        .spawn(move || {
            let crashed = supervise(factory, tokenizer, config, &rx);
            flag.store(false, Ordering::Release);
            // Anything still queued will never run.
            let why = crashed.map_or_else(
                || "the model's engine has stopped".to_string(),
                |why| format!("the model's engine stopped after repeated failures: {why}"),
            );
            while let Ok(job) = rx.try_recv() {
                let _ = job.events.send(Event::Failed(why.clone()));
            }
        })
        .expect("spawning the engine thread");
    EngineHandle { jobs, alive }
}

/// The text of a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "the model panicked".into())
}

/// Runs workers until the job channel closes (`None`), rebuilding the model
/// after each crash; gives up after [`MAX_RESTARTS`] crashes within
/// [`RESTART_WINDOW`], returning the last crash's reason.
fn supervise<M, F>(factory: F, tokenizer: Arc<dyn Tokenizer>, config: EngineConfig, jobs: &Receiver<Job>) -> Option<String>
where
    M: ServeModel + 'static,
    F: Fn() -> M,
{
    let mut crashes: VecDeque<Instant> = VecDeque::new();
    loop {
        let why = match catch_unwind(AssertUnwindSafe(&factory)) {
            Ok(model) => {
                let mut worker = Worker::new(model, tokenizer.clone(), config.clone());
                match catch_unwind(AssertUnwindSafe(|| worker.run(jobs))) {
                    Ok(()) => return None,
                    Err(payload) => {
                        let why = panic_message(payload.as_ref());
                        worker.fail_all(&why);
                        // The model may be in any state; dropping it must not
                        // take the supervisor down too.
                        let _ = catch_unwind(AssertUnwindSafe(move || drop(worker)));
                        why
                    }
                }
            }
            Err(payload) => format!("the model failed to build: {}", panic_message(payload.as_ref())),
        };
        let now = Instant::now();
        crashes.push_back(now);
        while crashes.front().is_some_and(|t| now.duration_since(*t) > RESTART_WINDOW) {
            crashes.pop_front();
        }
        if crashes.len() > MAX_RESTARTS {
            return Some(why);
        }
    }
}

/// Holds back text that might be the start of a stop string.
#[derive(Default)]
struct StopStrings {
    stops: Vec<String>,
    held: String,
}

impl StopStrings {
    /// Adds `text`; returns what can be released and the stop string that
    /// matched, if any (in which case nothing after it is released).
    fn push(&mut self, text: &str) -> (String, Option<String>) {
        self.held.push_str(text);
        let first = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|at| (at, s)))
            .min_by_key(|(at, _)| *at);
        if let Some((at, stop)) = first {
            let out = self.held[..at].to_string();
            let stop = stop.clone();
            self.held.clear();
            return (out, Some(stop));
        }
        let keep = self
            .stops
            .iter()
            .map(|s| longest_suffix_prefix(&self.held, s))
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, None)
    }

    fn flush(&mut self) -> String {
        core::mem::take(&mut self.held)
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `stop`,
/// on a character boundary.
fn longest_suffix_prefix(text: &str, stop: &str) -> usize {
    let max = text.len().min(stop.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&n| text.is_char_boundary(text.len() - n) && stop.as_bytes().starts_with(&text.as_bytes()[text.len() - n..]))
        .unwrap_or(0)
}

struct Sequence<S> {
    state: S,
    prompt: Vec<u32>,
    /// Tokens the state holds, in order.
    fed: Vec<u32>,
    prefilled: usize,
    /// The last sampled token, not yet fed.
    pending: Option<u32>,
    generated: usize,
    cached: usize,
    sampler: Sampler,
    decoder: StreamDecoder,
    stops: StopStrings,
    stop_tokens: Vec<u32>,
    max_tokens: usize,
    constraint: Option<Box<dyn Constraint>>,
    events: UnboundedSender<Event>,
    finished: Option<FinishReason>,
    /// Set when the sequence ended in an error rather than a finish.
    failed: Option<String>,
}

struct Worker<M: ServeModel> {
    model: M,
    tokenizer: Arc<dyn Tokenizer>,
    config: EngineConfig,
    cache: Option<PromptCache<M::State>>,
    active: Vec<Sequence<M::State>>,
}

impl<M: ServeModel> Worker<M> {
    fn new(model: M, tokenizer: Arc<dyn Tokenizer>, config: EngineConfig) -> Self {
        let cache = (config.prompt_cache_bytes > 0).then(|| PromptCache::new(config.prompt_cache_bytes));
        Self {
            model,
            tokenizer,
            config,
            cache,
            active: Vec::new(),
        }
    }

    /// Fails every active request with `why` (after a crash).
    fn fail_all(&mut self, why: &str) {
        for seq in self.active.drain(..) {
            let _ = seq.events.send(Event::Failed(format!("the model failed while running this request: {why}")));
        }
    }

    fn run(&mut self, jobs: &Receiver<Job>) {
        let mut open = true;
        loop {
            if self.active.is_empty() {
                if !open {
                    return;
                }
                match jobs.recv() {
                    Ok(job) => self.admit(job),
                    Err(_) => return,
                }
            }
            while open && self.active.len() < self.config.max_batch {
                match jobs.try_recv() {
                    Ok(job) => self.admit(job),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => open = false,
                }
            }
            self.active.retain(|s| !s.events.is_closed());
            self.prefill_step();
            self.decode_step();
            self.retire();
        }
    }

    fn admit(&mut self, job: Job) {
        let Job { request, events } = job;
        if request.prompt.is_empty() {
            let _ = events.send(Event::Error("the prompt is empty".into()));
            return;
        }
        if request.max_tokens == 0 {
            let _ = events.send(Event::Done {
                reason: FinishReason::Length,
                usage: Usage {
                    prompt_tokens: request.prompt.len(),
                    ..Usage::default()
                },
            });
            return;
        }
        let (state, reused) = match self.cache.as_mut().and_then(|c| c.lookup(&request.prompt)) {
            Some(hit) => (hit.state, hit.reused),
            None => (self.model.new_state(), 0),
        };
        let mut sampler = Sampler::new(request.sampling);
        for &t in &request.prompt {
            sampler.observe(t);
        }
        self.active.push(Sequence {
            state,
            fed: request.prompt[..reused].to_vec(),
            prefilled: reused,
            cached: reused,
            prompt: request.prompt,
            pending: None,
            generated: 0,
            sampler,
            decoder: StreamDecoder::new(),
            stops: StopStrings {
                stops: request.stop_strings.into_iter().filter(|s| !s.is_empty()).collect(),
                held: String::new(),
            },
            stop_tokens: request.stop_tokens,
            max_tokens: request.max_tokens,
            constraint: request.constraint,
            events,
            finished: None,
            failed: None,
        });
    }

    fn prefill_step(&mut self) {
        let chunk = self.config.prefill_chunk;
        for i in 0..self.active.len() {
            let seq = &mut self.active[i];
            if seq.finished.is_some() || seq.prefilled == seq.prompt.len() {
                continue;
            }
            let end = (seq.prefilled + chunk).min(seq.prompt.len());
            let tokens = seq.prompt[seq.prefilled..end].to_vec();
            let logits = self.model.forward(&mut seq.state, &tokens);
            seq.fed.extend_from_slice(&tokens);
            seq.prefilled = end;
            if end == seq.prompt.len() {
                let n = logits.dims()[0];
                let last = row(logits, n - 1);
                self.sample_into(i, last);
            }
        }
    }

    fn decode_step(&mut self) {
        let ready: Vec<usize> = (0..self.active.len())
            .filter(|&i| self.active[i].finished.is_none() && self.active[i].pending.is_some())
            .collect();
        if ready.is_empty() {
            return;
        }
        let tokens: Vec<u32> = ready.iter().map(|&i| self.active[i].pending.expect("filtered")).collect();
        let logits = {
            let mut states: Vec<&mut M::State> = Vec::with_capacity(ready.len());
            let mut rest: &mut [Sequence<M::State>] = &mut self.active;
            let mut base = 0;
            for &i in &ready {
                let (_, tail) = core::mem::take(&mut rest).split_at_mut(i - base);
                let (seq, after) = tail.split_first_mut().expect("index in range");
                states.push(&mut seq.state);
                rest = after;
                base = i + 1;
            }
            self.model.forward_batch(&mut states, &tokens)
        };
        for (r, &i) in ready.iter().enumerate() {
            let seq = &mut self.active[i];
            seq.fed.push(tokens[r]);
            seq.pending = None;
            let logits_row = row(logits.clone(), r);
            self.sample_into(i, logits_row);
        }
    }

    /// Samples the next token of sequence `i` from `logits` and handles it.
    fn sample_into(&mut self, i: usize, logits: Tensor<1>) {
        let tokenizer = self.tokenizer.clone();
        let seq = &mut self.active[i];
        let logits = match seq.constraint.as_ref() {
            Some(c) => {
                let allowed = c.allowed(&seq.stop_tokens);
                if !allowed.contains(&true) {
                    // The vocabulary cannot continue the constrained output;
                    // sampling now would break the constraint.
                    seq.failed = Some("no token of the vocabulary can continue the constrained output".into());
                    seq.finished = Some(FinishReason::Stop);
                    return;
                }
                let blocked: Vec<bool> = allowed.iter().map(|a| !a).collect();
                let vocab = logits.dims()[0];
                assert_eq!(blocked.len(), vocab, "the constraint covers a different vocabulary");
                let mask = Tensor::<1, Bool>::from_data(TensorData::new(blocked, [vocab]), &logits.device());
                logits.mask_fill(mask, f32::NEG_INFINITY)
            }
            None => logits,
        };
        let token = seq.sampler.sample(logits);
        seq.generated += 1;
        if let Some(c) = seq.constraint.as_mut()
            && !seq.stop_tokens.contains(&token)
        {
            c.advance(token);
        }

        if seq.stop_tokens.contains(&token) {
            finish(seq, tokenizer.as_ref(), FinishReason::Stop);
            return;
        }
        let text = seq.decoder.push(tokenizer.as_ref(), token);
        let (out, matched) = seq.stops.push(&text);
        if !out.is_empty() {
            let _ = seq.events.send(Event::Text(out));
        }
        if let Some(stop) = matched {
            seq.finished = Some(FinishReason::StopString(stop));
            return;
        }
        if seq.generated >= seq.max_tokens {
            finish(seq, tokenizer.as_ref(), FinishReason::Length);
            return;
        }
        seq.pending = Some(token);
    }

    fn retire(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].finished.is_none() {
                i += 1;
                continue;
            }
            let seq = self.active.swap_remove(i);
            let reason = seq.finished.clone().expect("checked");
            let _ = seq.events.send(match &seq.failed {
                Some(why) => Event::Failed(why.clone()),
                None => Event::Done {
                    reason,
                    usage: Usage {
                        prompt_tokens: seq.prompt.len(),
                        cached_tokens: seq.cached,
                        completion_tokens: seq.generated,
                    },
                },
            });
            if let Some(cache) = self.cache.as_mut() {
                let bytes = self.model.state_bytes(&seq.state);
                cache.insert(seq.fed, seq.state, bytes);
            }
        }
    }
}

/// Ends `seq`, releasing any text still held back.
fn finish<S>(seq: &mut Sequence<S>, tokenizer: &dyn Tokenizer, reason: FinishReason) {
    let tail = seq.decoder.flush(tokenizer);
    let (mut out, matched) = seq.stops.push(&tail);
    if let Some(stop) = matched {
        if !out.is_empty() {
            let _ = seq.events.send(Event::Text(out));
        }
        seq.finished = Some(FinishReason::StopString(stop));
        return;
    }
    out.push_str(&seq.stops.flush());
    if !out.is_empty() {
        let _ = seq.events.send(Event::Text(out));
    }
    seq.finished = Some(reason);
}

fn row(logits: Tensor<2>, i: usize) -> Tensor<1> {
    let vocab = logits.dims()[1];
    logits.narrow(0, i, 1).reshape([vocab])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_strings_hold_back_only_a_possible_start() {
        let mut s = StopStrings {
            stops: vec!["</end>".into()],
            held: String::new(),
        };
        assert_eq!(s.push("hello <"), ("hello ".into(), None));
        assert_eq!(s.push("/e"), (String::new(), None));
        assert_eq!(s.push("x"), ("</ex".into(), None), "not a stop after all");
        assert_eq!(s.push("a</end>tail"), ("a".into(), Some("</end>".into())));
    }

    #[test]
    fn the_earliest_stop_string_wins() {
        let mut s = StopStrings {
            stops: vec!["bb".into(), "ab".into()],
            held: String::new(),
        };
        assert_eq!(s.push("xabb"), ("x".into(), Some("ab".into())));
    }

    #[test]
    fn suffix_matching_respects_char_boundaries() {
        assert_eq!(longest_suffix_prefix("é<", "<x"), 1);
        assert_eq!(longest_suffix_prefix("aé", "éz"), 2);
        assert_eq!(longest_suffix_prefix("abc", "zz"), 0);
    }
}
