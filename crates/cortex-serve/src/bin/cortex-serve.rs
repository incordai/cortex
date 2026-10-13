//! `cortex-serve`: serves checkpoint directories over the OpenAI-, Anthropic-
//! and Ollama-compatible HTTP APIs.
//!
//! ```text
//! cortex-serve --model NAME=DIR [--model NAME=DIR ...] [--listen ADDR]
//!              [--max-loaded N] [--max-batch N] [--prefill-chunk N]
//!              [--prompt-cache-mb N]
//! ```
//!
//! Each `DIR` is a Hugging Face checkpoint directory (`config.json`,
//! safetensors, `tokenizer.json`, chat template). Models load on their first
//! request; at most `--max-loaded` stay resident, least recently used
//! unloaded first. The compute backend is the one this binary was built with
//! (`--features cuda`, `wgpu`, `flex`, ...).
//!
//! Calls: `cortex_serve::load_model` (`hf.rs`) per model, `Registry`,
//! `serve`.

use std::path::PathBuf;
use std::sync::Arc;

use cortex_core::tensor::Device;
use cortex_serve::{AppState, EngineConfig, Limits, Registry, load_model, serve};

const USAGE: &str = "usage: cortex-serve --model NAME=DIR [--model NAME=DIR ...] [--listen ADDR] \
[--max-loaded N] [--max-batch N] [--prefill-chunk N] [--prompt-cache-mb N]";

struct Options {
    listen: String,
    models: Vec<(String, PathBuf)>,
    max_loaded: usize,
    engine: EngineConfig,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        listen: "127.0.0.1:8080".into(),
        models: Vec::new(),
        max_loaded: 1,
        engine: EngineConfig::default(),
    };
    let mut args = args.peekable();
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let value = args.next().ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        let count = |v: &str| -> Result<usize, String> {
            v.parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("{flag} needs a positive whole number, got '{v}'"))
        };
        match flag.as_str() {
            "--listen" => options.listen = value,
            "--model" => {
                let (name, dir) = value
                    .split_once('=')
                    .filter(|(name, dir)| !name.is_empty() && !dir.is_empty())
                    .ok_or_else(|| format!("--model needs NAME=DIR, got '{value}'"))?;
                if options.models.iter().any(|(n, _)| n == name) {
                    return Err(format!("model name '{name}' is given twice"));
                }
                options.models.push((name.to_string(), PathBuf::from(dir)));
            }
            "--max-loaded" => options.max_loaded = count(&value)?,
            "--max-batch" => options.engine.max_batch = count(&value)?,
            "--prefill-chunk" => options.engine.prefill_chunk = count(&value)?,
            "--prompt-cache-mb" => {
                let mb = value
                    .parse::<usize>()
                    .map_err(|_| format!("--prompt-cache-mb needs a whole number, got '{value}'"))?;
                options.engine.prompt_cache_bytes = mb.saturating_mul(1 << 20);
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
    }
    if options.models.is_empty() {
        return Err(format!("no model given\n{USAGE}"));
    }
    for (name, dir) in &options.models {
        if !dir.join("config.json").is_file() {
            return Err(format!("model '{name}': {} has no config.json", dir.display()));
        }
    }
    Ok(options)
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let options = match parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(why) => {
            eprintln!("{why}");
            return std::process::ExitCode::from(2);
        }
    };

    let registry = Registry::new(options.max_loaded);
    for (name, dir) in options.models {
        let config = options.engine.clone();
        registry.add_lazy(
            name,
            Box::new(move || load_model(&dir, &Device::default(), config.clone())),
        );
    }
    let state = Arc::new(AppState {
        registry,
        limits: Limits::default(),
    });

    let listener = match tokio::net::TcpListener::bind(&options.listen).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("cannot listen on {}: {e}", options.listen);
            return std::process::ExitCode::FAILURE;
        }
    };
    eprintln!("cortex-serve listening on {}", options.listen);
    match serve(listener, state).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server stopped: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> impl Iterator<Item = String> {
        list.iter().map(|s| s.to_string()).collect::<Vec<_>>().into_iter()
    }

    #[test]
    fn bad_command_lines_are_refused_with_a_reason() {
        let err = |list: &[&str]| parse(args(list)).err().expect("refused");
        assert!(err(&[]).contains("no model"));
        assert!(err(&["--model", "x"]).contains("NAME=DIR"));
        assert!(err(&["--max-batch", "0", "--model", "a=b"]).contains("positive"));
        assert!(err(&["--frobnicate", "1"]).contains("unknown option"));
        assert!(err(&["--model"]).contains("needs a value"));
        assert!(err(&["--model", "a=/no/such/dir"]).contains("no config.json"));
    }
}
