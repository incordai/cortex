//! Tensor parallelism across the devices of one process: a single linear
//! layer split over several devices, the Megatron way.
//!
//! - [`ColumnParallelLinear`] splits `W[k, n]` by output columns. Each device
//!   computes its slice of the output; [`forward`](ColumnParallelLinear::forward)
//!   gathers the slices, [`forward_sharded`](ColumnParallelLinear::forward_sharded)
//!   leaves them where they are.
//! - [`RowParallelLinear`] splits `W[k, n]` by input rows. Each device
//!   computes a partial product from its slice of the input; the partials are
//!   summed on the output device.
//!
//! A column-parallel layer feeding a row-parallel one (an MLP's up and down
//! projections, attention's QKV and output projections) needs no gather in
//! between: the sharded output of the first is exactly the sharded input of
//! the second, so a block costs one reduction.
//!
//! Callers: model code. Calls: `Tensor::to_device`. Siblings: `placement.rs`
//! (pipeline parallelism, whole layers per device) and `distributed.rs`
//! (the same collectives across processes).

use alloc::vec::Vec;
use cortex_core::tensor::{Device, Tensor};

/// Splits `total` into `parts` contiguous sizes, earlier parts taking the
/// remainder.
fn split_sizes(total: usize, parts: usize) -> Vec<usize> {
    assert!(parts > 0 && total >= parts, "cannot split {total} over {parts} devices");
    (0..parts).map(|i| total / parts + usize::from(i < total % parts)).collect()
}

/// `x` on `device`, moved only if it lives elsewhere.
fn on(x: Tensor<2>, device: &Device) -> Tensor<2> {
    if &x.device() == device { x } else { x.to_device(device) }
}

/// A linear layer whose output columns are split over devices.
#[derive(Clone, Debug)]
pub struct ColumnParallelLinear {
    /// `(device, weight [k, n_i], bias [n_i])` per shard.
    shards: Vec<(Device, Tensor<2>, Option<Tensor<1>>)>,
    output: Device,
}

impl ColumnParallelLinear {
    /// Splits `weight [k, n]` (and `bias [n]`) as evenly as possible over
    /// `devices`; gathered outputs land on `output`.
    pub fn new(weight: Tensor<2>, bias: Option<Tensor<1>>, devices: &[Device], output: Device) -> Self {
        let [_, n] = weight.dims();
        let mut start = 0;
        let shards = split_sizes(n, devices.len())
            .into_iter()
            .zip(devices)
            .map(|(width, device)| {
                let w = weight.clone().narrow(1, start, width).to_device(device);
                let b = bias.clone().map(|b| b.narrow(0, start, width).to_device(device));
                start += width;
                (device.clone(), w, b)
            })
            .collect();
        Self { shards, output }
    }

    /// The per-device output slices `[m, n_i]`, each on its device.
    pub fn forward_sharded(&self, x: Tensor<2>) -> Vec<Tensor<2>> {
        self.shards
            .iter()
            .map(|(device, w, b)| {
                let y = on(x.clone(), device).matmul(w.clone());
                match b {
                    Some(b) => y + b.clone().unsqueeze_dim(0),
                    None => y,
                }
            })
            .collect()
    }

    /// The full output `[m, n]` on the output device.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let parts = self
            .forward_sharded(x)
            .into_iter()
            .map(|y| on(y, &self.output))
            .collect();
        Tensor::cat(parts, 1)
    }

    /// The devices, in shard order.
    pub fn devices(&self) -> Vec<Device> {
        self.shards.iter().map(|(d, _, _)| d.clone()).collect()
    }
}

/// A linear layer whose input rows are split over devices.
#[derive(Clone, Debug)]
pub struct RowParallelLinear {
    /// `(device, weight [k_i, n])` per shard.
    shards: Vec<(Device, Tensor<2>)>,
    widths: Vec<usize>,
    bias: Option<Tensor<1>>,
    output: Device,
}

impl RowParallelLinear {
    /// Splits `weight [k, n]` by rows over `devices`; the summed output (plus
    /// `bias [n]`, added once) lands on `output`.
    pub fn new(weight: Tensor<2>, bias: Option<Tensor<1>>, devices: &[Device], output: Device) -> Self {
        let [k, _] = weight.dims();
        let widths = split_sizes(k, devices.len());
        let mut start = 0;
        let shards = widths
            .iter()
            .zip(devices)
            .map(|(&rows, device)| {
                let w = weight.clone().narrow(0, start, rows).to_device(device);
                start += rows;
                (device.clone(), w)
            })
            .collect();
        let bias = bias.map(|b| b.to_device(&output));
        Self {
            shards,
            widths,
            bias,
            output,
        }
    }

    /// `Σ_i x_i · W_i` for inputs already split by column, `xs[i]` being
    /// `[m, k_i]` (for example a [`ColumnParallelLinear::forward_sharded`]
    /// output with matching splits).
    ///
    /// # Panics
    /// If the number or widths of the inputs do not match the shards.
    pub fn forward_sharded(&self, xs: Vec<Tensor<2>>) -> Tensor<2> {
        assert_eq!(xs.len(), self.shards.len(), "one input slice per shard");
        let mut sum: Option<Tensor<2>> = None;
        for ((device, w), (x, &width)) in self.shards.iter().zip(xs.into_iter().zip(&self.widths)) {
            assert_eq!(x.dims()[1], width, "input slice width does not match its shard");
            let partial = on(on(x, device).matmul(w.clone()), &self.output);
            sum = Some(match sum {
                Some(s) => s + partial,
                None => partial,
            });
        }
        let y = sum.expect("at least one shard");
        match &self.bias {
            Some(b) => y + b.clone().unsqueeze_dim(0),
            None => y,
        }
    }

    /// `x [m, k] · W + b` on the output device.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut start = 0;
        let xs = self
            .widths
            .iter()
            .map(|&w| {
                let slice = x.clone().narrow(1, start, w);
                start += w;
                slice
            })
            .collect();
        self.forward_sharded(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::split_sizes;

    #[test]
    fn splits_cover_the_total_in_order() {
        assert_eq!(split_sizes(10, 3), [4, 3, 3]);
        assert_eq!(split_sizes(4, 4), [1, 1, 1, 1]);
        assert_eq!(split_sizes(7, 1), [7]);
    }
}
