//! The models a server offers: loaded on first use, and unloaded least
//! recently used first when more than `max_loaded` are resident.
//!
//! Each model has its own slot and lock, and loading runs outside every
//! lock: a request for a resident model never waits behind another model's
//! load, and concurrent requests for a model being loaded wait for that one
//! load (single flight) and share its result, failure included. A model
//! whose engine stopped after repeated crashes is loaded afresh on its next
//! request.
//!
//! Unloading drops the model's [`EngineHandle`]; its worker thread exits once
//! in-flight requests finish, which frees the model and its device memory.
//!
//! Callers: the HTTP handlers in `api/` (`Registry::get` per request,
//! `Registry::list` for the model lists). Calls: the loaders supplied at
//! startup, which usually spawn an engine with `engine::spawn`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::constrain::TokenTable;
use crate::engine::EngineHandle;
use crate::model::{ChatTemplate, Tokenizer};

/// A model ready to serve.
pub struct ServedModel {
    /// Submits generations.
    pub engine: EngineHandle,
    /// The model's tokenizer.
    pub tokenizer: Arc<dyn Tokenizer>,
    /// Renders chats into prompts.
    pub template: Arc<dyn ChatTemplate>,
    /// Token bytes for constrained decoding, built once at load.
    pub tokens: Arc<TokenTable>,
}

impl ServedModel {
    /// Bundles a running engine with its tokenizer and template.
    pub fn new(engine: EngineHandle, tokenizer: Arc<dyn Tokenizer>, template: Arc<dyn ChatTemplate>) -> Self {
        let tokens = Arc::new(TokenTable::new(tokenizer.as_ref()));
        Self {
            engine,
            tokenizer,
            template,
            tokens,
        }
    }
}

/// Builds a model on demand.
pub type Loader = Box<dyn Fn() -> Result<ServedModel, String> + Send + Sync>;

/// Where one model stands.
enum State {
    Unloaded,
    /// A load is running; its waiters sleep on the slot's condvar.
    Loading,
    Loaded(Arc<ServedModel>),
}

struct SlotState {
    state: State,
    /// Loads completed so far, and the error of the latest if it failed: a
    /// waiter that saw a load start reports that load's failure.
    loads: u64,
    failure: Option<String>,
}

/// One served model.
struct Slot {
    name: String,
    loader: Option<Loader>,
    created: u64,
    last_used: AtomicU64,
    inner: Mutex<SlotState>,
    changed: Condvar,
}

impl Slot {
    fn lock(&self) -> MutexGuard<'_, SlotState> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn resident(&self) -> bool {
        matches!(self.lock().state, State::Loading | State::Loaded(_))
    }
}

/// Why a model could not be provided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// No model has this name.
    NotFound(String),
    /// The loader failed.
    LoadFailed(String),
}

/// The set of served models.
pub struct Registry {
    slots: Mutex<Vec<Arc<Slot>>>,
    clock: AtomicU64,
    max_loaded: usize,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Registry {
    /// An empty registry keeping at most `max_loaded` models resident.
    pub fn new(max_loaded: usize) -> Self {
        assert!(max_loaded > 0, "at least one model must fit");
        Self {
            slots: Mutex::new(Vec::new()),
            clock: AtomicU64::new(0),
            max_loaded,
        }
    }

    fn slots(&self) -> MutexGuard<'_, Vec<Arc<Slot>>> {
        self.slots.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn add(&self, name: String, loader: Option<Loader>, state: State) {
        self.slots().push(Arc::new(Slot {
            name,
            loader,
            created: now(),
            last_used: AtomicU64::new(0),
            inner: Mutex::new(SlotState {
                state,
                loads: 0,
                failure: None,
            }),
            changed: Condvar::new(),
        }));
    }

    /// Adds a model that is already loaded and is never unloaded.
    pub fn add_loaded(&self, name: impl Into<String>, model: ServedModel) {
        self.add(name.into(), None, State::Loaded(Arc::new(model)));
    }

    /// Adds a model built by `loader` the first time it is requested.
    pub fn add_lazy(&self, name: impl Into<String>, loader: Loader) {
        self.add(name.into(), Some(loader), State::Unloaded);
    }

    /// Names and creation times of every model, loaded or not.
    pub fn list(&self) -> Vec<(String, u64)> {
        self.slots().iter().map(|s| (s.name.clone(), s.created)).collect()
    }

    /// The model called `name`, loading it (and unloading the least
    /// recently used others) if needed. A request naming an unknown model is
    /// served by the only model when there is exactly one.
    ///
    /// Blocks while the model loads, so call this from a blocking context
    /// (`tokio::task::spawn_blocking`).
    pub fn get(&self, name: &str) -> Result<(String, Arc<ServedModel>), RegistryError> {
        let slot = {
            let slots = self.slots();
            match slots.iter().find(|s| s.name == name) {
                Some(s) => s.clone(),
                None if slots.len() == 1 => slots[0].clone(),
                None => return Err(RegistryError::NotFound(name.to_string())),
            }
        };
        slot.last_used.store(self.clock.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);

        let mut inner = slot.lock();
        let started = inner.loads;
        loop {
            match &inner.state {
                // A model whose engine gave up is loaded again (if it can be).
                State::Loaded(model) if model.engine.is_alive() || slot.loader.is_none() => {
                    return Ok((slot.name.clone(), model.clone()));
                }
                State::Loading => {
                    inner = slot.changed.wait(inner).unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                State::Unloaded if inner.loads > started => {
                    // The load this request waited for failed.
                    let why = inner.failure.clone().unwrap_or_else(|| "the model failed to load".into());
                    return Err(RegistryError::LoadFailed(why));
                }
                State::Unloaded | State::Loaded(_) => break,
            }
        }
        // This request loads the model, outside every lock.
        inner.state = State::Loading;
        drop(inner);

        self.make_room(&slot);
        let loader = slot.loader.as_ref().expect("an unloaded model has a loader");
        let loaded = loader();

        let mut inner = slot.lock();
        inner.loads += 1;
        let result = match loaded {
            Ok(model) => {
                let model = Arc::new(model);
                inner.state = State::Loaded(model.clone());
                inner.failure = None;
                Ok((slot.name.clone(), model))
            }
            Err(why) => {
                inner.state = State::Unloaded;
                inner.failure = Some(why.clone());
                Err(RegistryError::LoadFailed(why))
            }
        };
        drop(inner);
        slot.changed.notify_all();
        result
    }

    /// Unloads least recently used models until `loading` fits, so two large
    /// models are never in memory together. Pinned (eagerly added) models
    /// and models being loaded are never unloaded; if nothing else can go,
    /// the load proceeds beyond the limit rather than fail.
    fn make_room(&self, loading: &Arc<Slot>) {
        let slots: Vec<Arc<Slot>> = self.slots().clone();
        loop {
            let resident = slots.iter().filter(|s| s.resident()).count();
            if resident <= self.max_loaded {
                return;
            }
            let victim = slots
                .iter()
                .filter(|s| !Arc::ptr_eq(s, loading) && s.loader.is_some())
                .filter(|s| matches!(s.lock().state, State::Loaded(_)))
                .min_by_key(|s| s.last_used.load(Ordering::Relaxed));
            let Some(victim) = victim else {
                return;
            };
            let mut inner = victim.lock();
            if matches!(inner.state, State::Loaded(_)) {
                inner.state = State::Unloaded;
            }
        }
    }

    /// Names of the models currently resident (loaded or loading).
    pub fn resident(&self) -> Vec<String> {
        self.slots().iter().filter(|s| s.resident()).map(|s| s.name.clone()).collect()
    }
}
