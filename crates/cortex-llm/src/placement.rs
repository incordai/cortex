//! Pipeline parallelism: spreading a model's layers over several devices.
//!
//! A [`LayerPlacement`] assigns each layer index to a device. Loaders put a
//! layer's weights on [`LayerPlacement::device_of`] (for example through
//! `Checkpoint::tensor(name, placement.device_of(layer))`), and the forward
//! pass calls [`LayerPlacement::hop`] before each layer, which moves the
//! activations only when the layer lives on another device. A model too
//! large for one GPU can then run across two GPUs, or a GPU and the CPU.
//!
//! Callers: model code. Calls: `Tensor::to_device` (cortex-tensor). Sibling:
//! the collective ops in cortex-backend (`DistributedOps::all_reduce`), the
//! route for tensor parallelism, which splits single layers instead.

use alloc::vec::Vec;
use cortex_core::tensor::{Device, Tensor};

/// Which device each layer of a model lives on.
#[derive(Clone, Debug)]
pub struct LayerPlacement {
    /// One entry per stage: the device and the first layer it holds.
    stages: Vec<(Device, usize)>,
    layers: usize,
}

impl LayerPlacement {
    /// Every layer on `device`.
    pub fn single(device: Device, layers: usize) -> Self {
        Self {
            stages: alloc::vec![(device, 0)],
            layers,
        }
    }

    /// `counts[i]` consecutive layers on `devices[i]`.
    ///
    /// # Panics
    /// If the lists differ in length, are empty, or a count is zero.
    pub fn by_counts(devices: Vec<Device>, counts: &[usize]) -> Self {
        assert_eq!(devices.len(), counts.len(), "one layer count per device");
        assert!(!devices.is_empty(), "at least one device");
        assert!(counts.iter().all(|&c| c > 0), "every device must hold a layer");
        let mut first = 0;
        let stages = devices
            .into_iter()
            .zip(counts)
            .map(|(device, &count)| {
                let stage = (device, first);
                first += count;
                stage
            })
            .collect();
        Self { stages, layers: first }
    }

    /// `layers` split as evenly as possible over `devices`, earlier devices
    /// taking the remainder.
    pub fn even(devices: Vec<Device>, layers: usize) -> Self {
        let n = devices.len();
        assert!(n > 0 && layers >= n, "need at least one layer per device");
        let counts: Vec<usize> = (0..n).map(|i| layers / n + usize::from(i < layers % n)).collect();
        Self::by_counts(devices, &counts)
    }

    /// Number of layers placed.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The device holding `layer`.
    ///
    /// # Panics
    /// If `layer` is out of range.
    pub fn device_of(&self, layer: usize) -> &Device {
        assert!(layer < self.layers, "layer {layer} of {}", self.layers);
        let stage = self.stages.partition_point(|(_, first)| *first <= layer) - 1;
        &self.stages[stage].0
    }

    /// The device the model's output ends on (where the final norm and the
    /// output head belong).
    pub fn last_device(&self) -> &Device {
        &self.stages.last().expect("at least one stage").0
    }

    /// `x` on `layer`'s device: unchanged if it is already there, moved if
    /// the previous layer lived elsewhere.
    pub fn hop<const D: usize>(&self, x: Tensor<D>, layer: usize) -> Tensor<D> {
        let target = self.device_of(layer);
        if &x.device() == target {
            x
        } else {
            x.to_device(target)
        }
    }
}
