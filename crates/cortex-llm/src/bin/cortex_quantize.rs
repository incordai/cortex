//! `cortex-quantize`: convert a safetensors checkpoint to the packed
//! quantized format.
//!
//! ```text
//! cortex-quantize <input> <output-dir> [--bits 4] [--group 64]
//!                 [--mode affine|symmetric] [--mixed <high-bits>]
//! ```
//!
//! `<input>` is a `.safetensors` file or a checkpoint directory (sharded or
//! not). `--mixed` keeps the most sensitive projections at `<high-bits>`.
//! Calls `cortex_llm::checkpoint::quantize_checkpoint`.

use std::process::ExitCode;

use cortex_llm::checkpoint::{QuantRecipe, quantize_checkpoint};
use cortex_llm::quant::{QuantFormat, QuantMode};

struct Args {
    input: String,
    output: String,
    bits: u32,
    group: usize,
    mode: QuantMode,
    mixed: Option<u32>,
}

fn usage() -> &'static str {
    "usage: cortex-quantize <input> <output-dir> [--bits 4] [--group 64] \
     [--mode affine|symmetric] [--mixed <high-bits>]"
}

fn parse() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut bits = 4;
    let mut group = 64;
    let mut mode = QuantMode::Affine;
    let mut mixed = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--bits" => bits = value("--bits")?.parse().map_err(|e| format!("--bits: {e}"))?,
            "--group" => group = value("--group")?.parse().map_err(|e| format!("--group: {e}"))?,
            "--mixed" => mixed = Some(value("--mixed")?.parse().map_err(|e| format!("--mixed: {e}"))?),
            "--mode" => {
                mode = match value("--mode")?.as_str() {
                    "affine" => QuantMode::Affine,
                    "symmetric" => QuantMode::Symmetric,
                    other => return Err(format!("--mode must be affine or symmetric, got {other}")),
                }
            }
            "-h" | "--help" => return Err(usage().into()),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => positional.push(arg),
        }
    }
    let [input, output]: [String; 2] = positional
        .try_into()
        .map_err(|_| usage().to_string())?;
    Ok(Args { input, output, bits, group, mode, mixed })
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let recipe = (|| -> Result<QuantRecipe, String> {
        let low = QuantFormat::new(args.bits, args.group, args.mode).map_err(|e| e.to_string())?;
        Ok(match args.mixed {
            Some(high) => {
                let high = QuantFormat::new(high, args.group, args.mode).map_err(|e| e.to_string())?;
                QuantRecipe::mixed(low, high)
            }
            None => QuantRecipe::uniform(low),
        })
    })();
    let recipe = match recipe {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match quantize_checkpoint(&args.input, &args.output, &recipe) {
        Ok(report) => {
            println!(
                "quantized {} weights, kept {} tensors, wrote {:.1} MiB to {}",
                report.quantized,
                report.kept,
                report.bytes as f64 / (1024.0 * 1024.0),
                args.output
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
