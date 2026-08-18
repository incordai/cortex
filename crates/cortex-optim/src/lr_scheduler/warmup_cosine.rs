use cortex_core as cortex;

use super::{LrScheduler, LrSchedulerRecord, String};
use crate::LearningRate;
use crate::RecordState;
use crate::lr_scheduler::module_lr_scheduler::ModuleLrScheduler;
use cortex::config::Config;

/// The configuration for creating a [Warmup Cosine learning rate
/// scheduler](WarmupCosineLrScheduler).
///
/// This scheduler linearly warms up the learning rate from 0 to `initial_lr` over `warmup_steps`,
/// then decays it following a cosine curve to `min_lr` over the remaining steps.
///
/// This is the most commonly used schedule for transformer training.
#[derive(Config, Debug)]
pub struct WarmupCosineLrSchedulerConfig {
    /// The peak learning rate (reached after warmup).
    initial_lr: LearningRate,
    /// The final learning rate after cosine decay.
    #[config(default = 0.0)]
    min_lr: LearningRate,
    /// The number of warmup steps (linear ramp from 0 to initial_lr).
    warmup_steps: usize,
    /// The total number of training steps (including warmup).
    total_steps: usize,
}

impl WarmupCosineLrSchedulerConfig {
    /// Initializes a [Warmup Cosine learning rate scheduler](WarmupCosineLrScheduler).
    pub(crate) fn build(&self) -> Result<WarmupCosineLrScheduler, String> {
        if self.initial_lr <= 0. || self.initial_lr > 1. {
            return Err("Initial learning rate must be greater than 0 and at most 1".into());
        }
        if self.min_lr < 0.0 || self.min_lr > self.initial_lr {
            return Err(
                "Minimum learning rate must be at least 0 and at most equal to the initial \
                 learning rate"
                    .into(),
            );
        }
        if self.warmup_steps == 0 {
            return Err("Warmup steps must be at least 1".into());
        }
        if self.total_steps <= self.warmup_steps {
            return Err("Total steps must be greater than warmup steps".into());
        }

        Ok(WarmupCosineLrScheduler {
            initial_lr: self.initial_lr,
            min_lr: self.min_lr,
            warmup_steps: self.warmup_steps,
            total_steps: self.total_steps,
            current_step: usize::MAX,
        })
    }

    /// Initializes a [module learning rate scheduler](ModuleLrScheduler).
    ///
    /// # Errors
    ///
    /// An error will be returned if any of the following conditions is true:
    ///
    /// * `initial_lr` is out of range (0.0, 1.0]
    /// * `min_lr` is out of range [0.0, `initial_lr`]
    /// * `warmup_steps` is 0
    /// * `total_steps` is less than or equal to `warmup_steps`
    pub fn init(&self) -> Result<ModuleLrScheduler, String> {
        self.build().map(|s| s.into())
    }
}

/// A Warmup Cosine learning rate scheduler.
///
/// Linearly warms up from 0 to `initial_lr` over `warmup_steps`, then applies cosine annealing
/// from `initial_lr` to `min_lr` over the remaining steps.
///
/// After `total_steps`, the learning rate stays at `min_lr`.
///
/// See [WarmupCosineLrSchedulerConfig] for more information.
#[derive(Clone, Copy, Debug)]
pub struct WarmupCosineLrScheduler {
    initial_lr: LearningRate,
    min_lr: LearningRate,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LrScheduler for WarmupCosineLrScheduler {
    fn step(&mut self) -> LearningRate {
        // Overflow from usize::MAX to 0 on first call (same trick as CosineAnnealing)
        self.current_step = self.current_step.wrapping_add(1);

        if self.current_step < self.warmup_steps {
            // Linear warmup: lr = initial_lr * (step / warmup_steps)
            self.initial_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else if self.current_step >= self.total_steps {
            // Past total steps: clamp to min_lr
            self.min_lr
        } else {
            // Cosine decay phase
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress = self.current_step - self.warmup_steps;
            self.min_lr
                + 0.5
                    * (self.initial_lr - self.min_lr)
                    * (1.0
                        + (decay_progress as f64 / decay_steps as f64 * std::f64::consts::PI)
                            .cos())
        }
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::from_state(&WarmupCosineLrSchedulerState {
            current_step: self.current_step,
        })
    }

    fn load_record(&mut self, record: LrSchedulerRecord) {
        if let Some(state) = record.into_state::<WarmupCosineLrSchedulerState>() {
            self.current_step = state.current_step;
        }
    }
}

/// The serializable state of a [warmup cosine scheduler](WarmupCosineLrScheduler).
#[derive(RecordState, Clone, Debug)]
pub struct WarmupCosineLrSchedulerState {
    current_step: usize,
}

#[cfg(test)]
mod tests {
    use super::super::test_utils;
    use super::*;

    #[test]
    fn config_initial_lr_too_low() {
        let r = WarmupCosineLrSchedulerConfig::new(0., 10, 100).build();
        assert!(r.is_err());
    }

    #[test]
    fn config_initial_lr_too_high() {
        let r = WarmupCosineLrSchedulerConfig::new(1.5, 10, 100).build();
        assert!(r.is_err());
    }

    #[test]
    fn config_warmup_zero() {
        let r = WarmupCosineLrSchedulerConfig::new(0.5, 0, 100).build();
        assert!(r.is_err());
    }

    #[test]
    fn config_total_too_small() {
        let r = WarmupCosineLrSchedulerConfig::new(0.5, 10, 10).build();
        assert!(r.is_err());
    }

    #[test]
    fn test_warmup_phase() {
        const LR: LearningRate = 1.0;

        let scheduler = WarmupCosineLrSchedulerConfig::new(LR, 4, 8).build().unwrap();
        // Steps 0,1,2,3 are warmup: lr = LR * step/4
        let expected_lrs = [
            0.0,  // step 0: 1.0 * 0/4
            0.25, // step 1: 1.0 * 1/4
            0.5,  // step 2: 1.0 * 2/4
            0.75, // step 3: 1.0 * 3/4
        ];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_cosine_phase() {
        const LR: LearningRate = 1.0;
        const MIN_LR: LearningRate = 0.0;

        let mut scheduler = WarmupCosineLrSchedulerConfig::new(LR, 2, 6)
            .with_min_lr(MIN_LR)
            .build()
            .unwrap();

        // Consume warmup steps (0, 1)
        let _ = scheduler.step(); // step 0
        let _ = scheduler.step(); // step 1

        // Cosine phase: steps 2,3,4,5 map to decay_progress 0,1,2,3 out of decay_steps=4
        let lr2 = scheduler.step(); // step 2: cos(0) = 1.0, lr = 0 + 0.5*1*(1+1) = 1.0
        let lr3 = scheduler.step(); // step 3: cos(pi/4), lr = 0.5*(1+cos(pi/4))
        let lr4 = scheduler.step(); // step 4: cos(pi/2) = 0, lr = 0.5*(1+0) = 0.5
        let lr5 = scheduler.step(); // step 5: cos(3pi/4), lr = 0.5*(1+cos(3pi/4))

        assert!((lr2 - 1.0).abs() < 1e-10);
        assert!((lr4 - 0.5).abs() < 1e-10);
        // lr3 > lr4 > lr5 (monotonically decreasing)
        assert!(lr3 > lr4);
        assert!(lr4 > lr5);
    }

    #[test]
    fn test_clamp_after_total() {
        const LR: LearningRate = 1.0;
        const MIN_LR: LearningRate = 0.1;

        let mut scheduler = WarmupCosineLrSchedulerConfig::new(LR, 2, 4)
            .with_min_lr(MIN_LR)
            .build()
            .unwrap();

        // Consume all steps
        for _ in 0..4 {
            scheduler.step();
        }

        // Beyond total_steps should clamp to min_lr
        let lr = scheduler.step();
        assert!((lr - MIN_LR).abs() < 1e-10);
    }

    #[test]
    fn test_save_and_load() {
        let scheduler = WarmupCosineLrSchedulerConfig::new(1.0, 5, 20)
            .build()
            .unwrap();
        test_utils::check_save_load(scheduler, 7);
    }
}
