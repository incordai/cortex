//! The packed quantized weight format and its host-side reference.
//!
//! # Layout
//!
//! A weight `W[k, n]` (row-major, `y = x · W`) is stored row by row. Each row
//! is a little-endian bit stream of `n` unsigned codes of `bits` bits, so
//! column `j` occupies bits `j * bits .. (j + 1) * bits` of its row. Rows are
//! padded to whole `u32` words; because `n` is a multiple of 32, every 32
//! columns occupy exactly `bits` words and no padding is ever needed.
//!
//! Each row carries one scale per `group_size` consecutive columns
//! (`scales[k, n / group_size]`), and in [`QuantMode::Affine`] one bias per
//! group as well:
//!
//! - [`QuantMode::Affine`]: `w = scale * q + bias`, `q in 0..2^bits`.
//! - [`QuantMode::Symmetric`]: `w = scale * (q - 2^(bits-1))`; the code is the
//!   signed value in offset binary, so this is affine with the implicit bias
//!   `-scale * 2^(bits-1)`.
//!
//! At 4 bits with groups of 32 this is bit-for-bit the layout of a
//! block-32 symmetric Q4 weight with nibbles packed low-first along `n`, once
//! two's-complement nibbles are converted to offset binary (`v ^ 8`).
//!
//! # Wiring
//!
//! Producers: [`quantize`] (here), and any loader that already holds packed
//! data in this layout. Consumers: [`dequantize`] (host reference), the
//! backend-generic default in `ops.rs` (`LlmOps::quantized_matmul`), and the
//! CubeCL kernels in `kernels/qmm.rs`, which must decode codes exactly as
//! [`code_at`] does.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// How codes map back to real values. See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// `w = scale * (q - 2^(bits-1))`: one scale per group, no stored bias.
    Symmetric,
    /// `w = scale * q + bias`: one scale and one bias per group.
    Affine,
}

/// A packed quantization format: code width, group width and mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QuantFormat {
    bits: u32,
    group_size: usize,
    mode: QuantMode,
}

/// Why a format or a weight shape was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantError {
    /// `bits` must be one of 2, 3, 4, 5, 6 or 8.
    UnsupportedBits(u32),
    /// `group_size` must be a multiple of 8 between 8 and 256.
    UnsupportedGroupSize(usize),
    /// `n` must be a multiple of both 32 and the group size.
    ColumnsNotAligned {
        /// Number of columns of the weight.
        n: usize,
        /// The group size it was checked against.
        group_size: usize,
    },
    /// Packed data must be held as `u32` words.
    DataNotU32,
    /// A buffer's length does not match the shape it was declared with.
    LengthMismatch {
        /// Which buffer.
        what: &'static str,
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBits(b) => write!(f, "unsupported code width {b} (use 2, 3, 4, 5, 6 or 8)"),
            Self::UnsupportedGroupSize(g) => {
                write!(f, "unsupported group size {g} (use a multiple of 8 in 8..=256)")
            }
            Self::ColumnsNotAligned { n, group_size } => write!(
                f,
                "{n} columns is not a multiple of 32 and of the group size {group_size}"
            ),
            Self::DataNotU32 => write!(f, "packed data must be a u32 tensor"),
            Self::LengthMismatch {
                what,
                expected,
                actual,
            } => write!(f, "{what} holds {actual} elements, expected {expected}"),
        }
    }
}

impl QuantFormat {
    /// A format, validated.
    ///
    /// Group sizes are multiples of 8 because the GPU kernels give each thread
    /// eight adjacent columns and read one scale for all eight.
    pub fn new(bits: u32, group_size: usize, mode: QuantMode) -> Result<Self, QuantError> {
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(QuantError::UnsupportedBits(bits));
        }
        if group_size == 0 || !group_size.is_multiple_of(8) || group_size > 256 {
            return Err(QuantError::UnsupportedGroupSize(group_size));
        }
        Ok(Self {
            bits,
            group_size,
            mode,
        })
    }

    /// Code width in bits.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Columns sharing one scale (and bias).
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// How codes map back to values.
    pub fn mode(&self) -> QuantMode {
        self.mode
    }

    /// Checks that a weight with `n` columns can be stored in this format.
    pub fn check_columns(&self, n: usize) -> Result<(), QuantError> {
        if n == 0 || !n.is_multiple_of(32) || !n.is_multiple_of(self.group_size) {
            return Err(QuantError::ColumnsNotAligned {
                n,
                group_size: self.group_size,
            });
        }
        Ok(())
    }

    /// `u32` words per row for `n` columns (`n * bits / 32`).
    pub fn words_per_row(&self, n: usize) -> usize {
        n / 32 * self.bits as usize
    }

    /// Scale groups per row for `n` columns.
    pub fn groups_per_row(&self, n: usize) -> usize {
        n / self.group_size
    }

    /// The largest code, `2^bits - 1`.
    pub fn max_code(&self) -> u32 {
        (1u32 << self.bits) - 1
    }

    /// The offset subtracted from codes in [`QuantMode::Symmetric`].
    pub fn symmetric_offset(&self) -> u32 {
        1u32 << (self.bits - 1)
    }
}

/// A weight `[k, n]` packed in a [`QuantFormat`], on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedWeight {
    /// Rows (input features).
    pub k: usize,
    /// Columns (output features).
    pub n: usize,
    /// The format the codes are packed in.
    pub format: QuantFormat,
    /// `[k, words_per_row(n)]` packed codes.
    pub data: Vec<u32>,
    /// `[k, groups_per_row(n)]` scales.
    pub scales: Vec<f32>,
    /// `[k, groups_per_row(n)]` biases, present exactly in affine mode.
    pub biases: Option<Vec<f32>>,
}

impl PackedWeight {
    /// Wraps existing packed buffers, checking every length against the shape.
    pub fn from_parts(
        k: usize,
        n: usize,
        format: QuantFormat,
        data: Vec<u32>,
        scales: Vec<f32>,
        biases: Option<Vec<f32>>,
    ) -> Result<Self, QuantError> {
        format.check_columns(n)?;
        check_len("data", k * format.words_per_row(n), data.len())?;
        check_len("scales", k * format.groups_per_row(n), scales.len())?;
        match (format.mode, &biases) {
            (QuantMode::Affine, Some(b)) => check_len("biases", scales.len(), b.len())?,
            (QuantMode::Affine, None) => check_len("biases", scales.len(), 0)?,
            (QuantMode::Symmetric, Some(b)) => check_len("biases", 0, b.len())?,
            (QuantMode::Symmetric, None) => {}
        }
        Ok(Self {
            k,
            n,
            format,
            data,
            scales,
            biases,
        })
    }
}

fn check_len(what: &'static str, expected: usize, actual: usize) -> Result<(), QuantError> {
    if expected == actual {
        Ok(())
    } else {
        Err(QuantError::LengthMismatch {
            what,
            expected,
            actual,
        })
    }
}

/// The code of column `j` in a packed row.
///
/// The reference decoder: the kernels must agree with it bit for bit. A code
/// may straddle two words when `bits` does not divide 32 (3, 5, 6).
pub fn code_at(row: &[u32], j: usize, bits: u32) -> u32 {
    let pos = j * bits as usize;
    let word = pos / 32;
    let offset = (pos % 32) as u32;
    let mask = (1u32 << bits) - 1;
    let mut v = row[word] >> offset;
    if offset + bits > 32 {
        v |= row[word + 1] << (32 - offset);
    }
    v & mask
}

fn put_code(row: &mut [u32], j: usize, bits: u32, code: u32) {
    let pos = j * bits as usize;
    let word = pos / 32;
    let offset = (pos % 32) as u32;
    row[word] |= code << offset;
    if offset + bits > 32 {
        row[word + 1] |= code >> (32 - offset);
    }
}

/// Quantizes a row-major `[k, n]` weight with round-to-nearest.
///
/// Affine groups map `[min, max]` onto `0..=2^bits-1`; symmetric groups map
/// `[-absmax, absmax]` onto `-(2^(bits-1)-1) ..= 2^(bits-1)-1`, leaving the
/// most negative code unused so zero stays exactly representable.
pub fn quantize(w: &[f32], k: usize, n: usize, format: QuantFormat) -> Result<PackedWeight, QuantError> {
    format.check_columns(n)?;
    check_len("weight", k * n, w.len())?;

    let wpr = format.words_per_row(n);
    let gpr = format.groups_per_row(n);
    let g = format.group_size;
    let bits = format.bits;
    let mut data = vec![0u32; k * wpr];
    let mut scales = vec![0f32; k * gpr];
    let mut biases = match format.mode {
        QuantMode::Affine => Some(vec![0f32; k * gpr]),
        QuantMode::Symmetric => None,
    };

    for r in 0..k {
        let row = &w[r * n..(r + 1) * n];
        let packed = &mut data[r * wpr..(r + 1) * wpr];
        for group in 0..gpr {
            let values = &row[group * g..(group + 1) * g];
            let (scale, bias) = match format.mode {
                QuantMode::Affine => {
                    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = (hi - lo) / format.max_code() as f32;
                    (scale, lo)
                }
                QuantMode::Symmetric => {
                    let absmax = values.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let levels = (format.symmetric_offset() - 1).max(1) as f32;
                    let scale = absmax / levels;
                    (scale, -scale * format.symmetric_offset() as f32)
                }
            };
            scales[r * gpr + group] = scale;
            if let Some(b) = biases.as_mut() {
                b[r * gpr + group] = bias;
            }
            for (i, &v) in values.iter().enumerate() {
                let code = if scale > 0.0 {
                    ((v - bias) / scale).round().clamp(0.0, format.max_code() as f32) as u32
                } else {
                    // A constant group: every value equals the bias (affine)
                    // or zero (symmetric), so the code that decodes to it is 0
                    // or the symmetric offset respectively.
                    match format.mode {
                        QuantMode::Affine => 0,
                        QuantMode::Symmetric => format.symmetric_offset(),
                    }
                };
                put_code(packed, group * g + i, bits, code);
            }
        }
    }

    PackedWeight::from_parts(k, n, format, data, scales, biases)
}

/// Decodes a packed weight back to row-major `[k, n]` values.
pub fn dequantize(p: &PackedWeight) -> Vec<f32> {
    let wpr = p.format.words_per_row(p.n);
    let gpr = p.format.groups_per_row(p.n);
    let g = p.format.group_size;
    let offset = p.format.symmetric_offset() as f32;
    let mut out = vec![0f32; p.k * p.n];
    for r in 0..p.k {
        let row = &p.data[r * wpr..(r + 1) * wpr];
        for j in 0..p.n {
            let group = r * gpr + j / g;
            let scale = p.scales[group];
            let code = code_at(row, j, p.format.bits) as f32;
            out[r * p.n + j] = match &p.biases {
                Some(b) => scale * code + b[group],
                None => scale * (code - offset),
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Every code survives a pack/unpack round trip, including the 3-, 5- and
    /// 6-bit widths whose codes straddle word boundaries.
    #[test]
    fn codes_round_trip_at_every_width() {
        for bits in [2u32, 3, 4, 5, 6, 8] {
            let n = 96;
            let words = n / 32 * bits as usize;
            let mut row = vec![0u32; words];
            let codes: Vec<u32> = (0..n).map(|j| (j as u32 * 7 + 3) & ((1 << bits) - 1)).collect();
            for (j, &c) in codes.iter().enumerate() {
                put_code(&mut row, j, bits, c);
            }
            for (j, &c) in codes.iter().enumerate() {
                assert_eq!(code_at(&row, j, bits), c, "bits {bits} column {j}");
            }
        }
    }

    /// Reconstruction error is bounded by half a quantization step per group.
    #[test]
    fn dequantize_is_within_half_a_step() {
        let (k, n) = (5, 128);
        let w = noise(k * n, 11);
        for bits in [2u32, 3, 4, 5, 6, 8] {
            for mode in [QuantMode::Affine, QuantMode::Symmetric] {
                for g in [32usize, 64, 128] {
                    let f = QuantFormat::new(bits, g, mode).unwrap();
                    let p = quantize(&w, k, n, f).unwrap();
                    let back = dequantize(&p);
                    for r in 0..k {
                        for j in 0..n {
                            let step = p.scales[r * f.groups_per_row(n) + j / g];
                            let err = (back[r * n + j] - w[r * n + j]).abs();
                            assert!(
                                err <= step * 0.5 + 1e-6,
                                "bits {bits} {mode:?} g{g}: |{} - {}| = {err} > step/2 {}",
                                back[r * n + j],
                                w[r * n + j],
                                step * 0.5
                            );
                        }
                    }
                }
            }
        }
    }

    /// The 4-bit symmetric group-32 layout is the block-32 Q4 layout with
    /// nibbles low-first: column `j` is nibble `j % 8` of word `j / 8`.
    #[test]
    fn four_bit_symmetric_matches_the_nibble_layout() {
        let f = QuantFormat::new(4, 32, QuantMode::Symmetric).unwrap();
        let w = noise(64, 5);
        let p = quantize(&w, 1, 64, f).unwrap();
        for j in 0..64 {
            let nibble = (p.data[j / 8] >> (4 * (j % 8))) & 0xF;
            assert_eq!(nibble, code_at(&p.data, j, 4));
        }
    }

    #[test]
    fn constant_groups_decode_exactly() {
        let w = vec![0.25f32; 64];
        for mode in [QuantMode::Affine, QuantMode::Symmetric] {
            let f = QuantFormat::new(4, 32, mode).unwrap();
            let back = dequantize(&quantize(&w, 1, 64, f).unwrap());
            for v in back {
                assert!((v - 0.25).abs() < 1e-6, "{mode:?}: {v}");
            }
        }
        let zeros = vec![0f32; 32];
        for mode in [QuantMode::Affine, QuantMode::Symmetric] {
            let f = QuantFormat::new(3, 32, mode).unwrap();
            assert!(dequantize(&quantize(&zeros, 1, 32, f).unwrap()).iter().all(|v| *v == 0.0));
        }
    }

    #[test]
    fn bad_formats_and_shapes_are_refused() {
        assert_eq!(QuantFormat::new(7, 32, QuantMode::Affine), Err(QuantError::UnsupportedBits(7)));
        assert_eq!(
            QuantFormat::new(4, 12, QuantMode::Affine),
            Err(QuantError::UnsupportedGroupSize(12))
        );
        let f = QuantFormat::new(4, 64, QuantMode::Affine).unwrap();
        assert!(matches!(f.check_columns(96), Err(QuantError::ColumnsNotAligned { .. })));
        assert!(matches!(
            PackedWeight::from_parts(1, 64, f, vec![0; 7], vec![0.0; 1], Some(vec![0.0; 1])),
            Err(QuantError::LengthMismatch { what: "data", .. })
        ));
    }
}
