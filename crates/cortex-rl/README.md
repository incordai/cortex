# Cortex RL

<!-- This crate should be used with [burn](https://github.com/tracel-ai/burn). -->

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-rl.svg)](https://crates.io/crates/burn-rl)
[![Documentation](https://docs.rs/burn-rl/badge.svg)](https://docs.rs/burn-rl)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- `Environment`: a simulation an agent acts in, returning a `StepResult` per action.
- `Policy`: maps observations to actions; `PolicyLearner` updates a policy from experience.
- `TransitionBuffer`: a replay buffer of `Transition`s sampled in batches.

## Usage

`cortex-rl` is not a default dependency of `cortex`. Enable the `rl` feature, together with `train` for
the RL learner:

```toml
cortex = { version = "0.22", features = ["train", "rl", "flex"] }
```

The crate is re-exported as `cortex::rl`. See the
[DQN agent example](https://github.com/tracel-ai/burn/tree/main/examples/dqn-agent).

<!-- cortex-crate-footer -->

---

Part of the [Cortex](https://github.com/tracel-ai/burn) deep learning framework. See the
[Cortex Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
