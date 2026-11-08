# Cortex RL

> Reinforcement learning building blocks for [Cortex](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-rl.svg)](https://crates.io/crates/cortex-rl)
[![Documentation](https://docs.rs/cortex-rl/badge.svg)](https://docs.rs/cortex-rl)
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
[Cortex Book](https://cortex.dev/books/cortex/) and the [API documentation](https://docs.rs/cortex).
