//! minq CLI: `run` (generate text), `quantize` (safetensors -> .minq),
//! `bench` (prefill/decode throughput).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safetensors::SafeTensors;

use minq::engine::Engine;
use minq::format::WeightTensor;
use minq::model::{self, Model};
use minq::quantize::{QuantDtype, QuantizedTensor, BLOCK_SIZE};
use minq::sampler::Sampler;
use minq::tokenizer::TextTokenizer;
use minq::Result;

#[derive(Parser)]
#[command(
    name = "minq",
    version,
    about = "A from-scratch CPU inference engine for LLaMA-family transformers"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text from a prompt.
    Run {
        /// Model path: a .minq file, a .safetensors file, or a model directory.
        #[arg(long)]
        model: PathBuf,
        /// Path to tokenizer.json.
        #[arg(long)]
        tokenizer: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Sampling temperature; 0 means greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Top-k filter; 0 disables it.
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Top-p (nucleus) filter.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// RNG seed for sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Number of rayon worker threads (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Export a safetensors model into the quantized .minq format.
    Quantize {
        /// Input .safetensors file or model directory (with config.json).
        #[arg(long)]
        input: PathBuf,
        /// Output .minq file.
        #[arg(long)]
        output: PathBuf,
        /// Quantization dtype: q8_0 or q4_0.
        #[arg(long)]
        dtype: String,
    },
    /// Measure prefill and decode throughput.
    Bench {
        /// Model path (.minq, .safetensors file, or directory).
        #[arg(long)]
        model: PathBuf,
        /// Number of random prompt tokens for the prefill phase.
        #[arg(long, default_value_t = 128)]
        prompt_tokens: usize,
        /// Number of tokens to generate in the decode phase.
        #[arg(long, default_value_t = 64)]
        gen_tokens: usize,
        /// Number of rayon worker threads (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Evaluate perplexity of a text file under the model.
    EvalPpl {
        /// Model path (.minq, .safetensors file, or directory).
        #[arg(long)]
        model: PathBuf,
        /// Path to tokenizer.json.
        #[arg(long)]
        tokenizer: PathBuf,
        /// Plain-text input file.
        #[arg(long)]
        input: PathBuf,
        /// Non-overlapping context window length.
        #[arg(long, default_value_t = 512)]
        context: usize,
        /// Keep only the first N tokens of the encoded input.
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Number of rayon worker threads (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            threads,
        } => cmd_run(
            &model,
            &tokenizer,
            &prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            threads,
        ),
        Command::Quantize {
            input,
            output,
            dtype,
        } => cmd_quantize(&input, &output, &dtype),
        Command::Bench {
            model,
            prompt_tokens,
            gen_tokens,
            threads,
        } => cmd_bench(&model, prompt_tokens, gen_tokens, threads),
        Command::EvalPpl {
            model,
            tokenizer,
            input,
            context,
            max_tokens,
            threads,
        } => cmd_eval_ppl(&model, &tokenizer, &input, context, max_tokens, threads),
    }
}

/// Load a model from any supported path. `.minfer` is the legacy extension
/// of the same format and is still accepted.
fn load_model(path: &Path) -> Result<Model> {
    if path
        .extension()
        .is_some_and(|ext| ext == "minq" || ext == "minfer")
    {
        model::load_minq(path)
    } else {
        model::load_safetensors(path)
    }
}

fn set_threads(threads: Option<usize>) -> Result<()> {
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| minq::MinqError::Model(format!("rayon init: {e}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    threads: Option<usize>,
) -> Result<()> {
    set_threads(threads)?;
    let model = load_model(model_path)?;
    let tokenizer = TextTokenizer::from_file(tokenizer_path)?;
    let prompt_ids = tokenizer.encode(prompt, true)?;
    if prompt_ids.is_empty() {
        return Err(minq::MinqError::Model(
            "prompt encodes to zero tokens".into(),
        ));
    }

    let sampler = Sampler::new(temperature, top_k, top_p, seed);
    let mut engine = Engine::new(model, sampler);
    engine.stop_tokens = tokenizer.eos_token_ids();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    // Incremental detokenization: re-decode all generated ids each step and
    // print only the new suffix, so multi-byte UTF-8 characters split across
    // tokens never reach the terminal as replacement characters.
    let mut detok = minq::tokenizer::IncrementalDecode::new();
    let stats = engine.generate(&prompt_ids, max_tokens, |token| {
        if let Some(piece) = detok.push(&tokenizer, token) {
            // Propagate IO errors (e.g. broken pipe from `| head`) instead of
            // swallowing them; generation aborts and main exits with the error.
            out.write_all(piece.as_bytes())?;
            out.flush()?;
        }
        Ok(())
    })?;
    out.write_all(b"\n")?;

    eprintln!(
        "\n[prefill: {} tokens in {:.2}s ({:.1} tok/s) | decode: {} tokens in {:.2}s ({:.1} tok/s)]",
        stats.prompt_tokens,
        stats.prefill_secs,
        stats.prefill_tokens_per_sec(),
        stats.gen_tokens,
        stats.decode_secs,
        stats.decode_tokens_per_sec(),
    );
    Ok(())
}

fn cmd_quantize(input: &Path, output: &Path, dtype: &str) -> Result<()> {
    let dtype = QuantDtype::parse(dtype)?;
    let (dir, files) = model::resolve_safetensors(input)?;
    let mut config = model::hf_config(&dir)?;
    config.dtype = dtype.name().to_string();

    // Names of every tensor the model needs, in deterministic order.
    let mut names: Vec<String> = vec!["model.embed_tokens.weight".to_string()];
    for i in 0..config.n_layers {
        let p = format!("model.layers.{i}");
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("{p}.{suffix}"));
        }
    }
    names.push("model.norm.weight".to_string());
    names.push("lm_head.weight".to_string());

    // Pre-scan shard headers (only the small JSON header is read, never the
    // weights) so we can validate the plan up front and write records in
    // deterministic sorted-name order.
    let mut location: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (idx, file) in files.iter().enumerate() {
        for name in model::scan_safetensors_names(file)? {
            location.insert(name, idx);
        }
    }

    // Q/K/V biases, Qwen3 q/k-norms and a tied LM head are legitimately absent.
    let mut present: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for name in names {
        if location.contains_key(&name) {
            present.push(name);
        } else {
            let optional = name.ends_with(".bias")
                || name.ends_with("q_norm.weight")
                || name.ends_with("k_norm.weight")
                || name == "lm_head.weight";
            if !optional {
                missing.push(name);
            }
        }
    }
    if !missing.is_empty() {
        return Err(minq::MinqError::Model(format!(
            "missing required tensors: {}",
            missing.join(", ")
        )));
    }
    config.has_qkv_bias = location.contains_key("model.layers.0.self_attn.q_proj.bias");
    present.sort();

    // True streaming export: write the header up front, then convert,
    // quantize and immediately write one tensor at a time. Peak memory is
    // one shard buffer plus one tensor, never the whole packed model.
    let mut out = std::io::BufWriter::new(fs::File::create(output)?);
    minq::format::write_minq_header(&mut out, &config)?;
    let mut raw_bytes = 0usize;
    let mut packed_bytes = 0usize;
    let mut cached: Option<(usize, Vec<u8>)> = None;
    for name in &present {
        let shard = location[name];
        if cached.as_ref().is_none_or(|(i, _)| *i != shard) {
            cached = Some((shard, fs::read(&files[shard])?));
        }
        let st = SafeTensors::deserialize(&cached.as_ref().unwrap().1)?;
        let t = model::st_to_tensor(&st, name)?;
        raw_bytes += t.numel() * 4;
        let record = quantize_tensor(name, t, dtype)?;
        packed_bytes += match &record.1 {
            WeightTensor::F32(t) => t.numel() * 4,
            WeightTensor::Quant(q) => q.data.len(),
        };
        minq::format::write_minq_record(&mut out, &record.0, &record.1)?;
    }
    std::io::Write::flush(&mut out)?;
    eprintln!(
        "wrote {} ({} tensors, {:.1} MB f32 -> {:.1} MB packed, dtype {})",
        output.display(),
        present.len(),
        raw_bytes as f64 / 1e6,
        packed_bytes as f64 / 1e6,
        dtype.name(),
    );
    Ok(())
}

/// Quantize 2-D weight matrices (except the embedding table, which is only
/// ever indexed); keep norms, biases and odd-shaped tensors in f32.
fn quantize_tensor(name: &str, t: minq::tensor::Tensor, dtype: QuantDtype) -> Result<(String, WeightTensor)> {
    let quantizable = name.ends_with(".weight")
        && !name.contains("embed_tokens")
        && t.ndim() == 2
        && t.shape[1] % BLOCK_SIZE == 0;
    if quantizable {
        Ok((name.to_string(), WeightTensor::Quant(QuantizedTensor::from_tensor(&t, dtype)?)))
    } else {
        Ok((name.to_string(), WeightTensor::F32(t)))
    }
}

fn cmd_bench(model_path: &Path, prompt_tokens: usize, gen_tokens: usize, threads: Option<usize>) -> Result<()> {
    set_threads(threads)?;
    let model = load_model(model_path)?;
    let vocab = model.config.vocab_size;

    let mut rng = StdRng::seed_from_u64(0);
    let prompt: Vec<u32> = (0..prompt_tokens.max(1))
        .map(|_| rng.gen_range(0..vocab as u32))
        .collect();

    let mut engine = Engine::new(model, Sampler::greedy());
    let stats = engine.generate(&prompt, gen_tokens, |_| Ok(()))?;

    println!("model:      {}", model_path.display());
    println!("dtype:      {}", engine.model.config.dtype);
    println!(
        "threads:    {}",
        threads.unwrap_or_else(rayon::current_num_threads)
    );
    println!(
        "prefill:    {} tokens in {:.3}s -> {:.1} tok/s",
        stats.prompt_tokens,
        stats.prefill_secs,
        stats.prefill_tokens_per_sec()
    );
    println!(
        "decode:     {} tokens in {:.3}s -> {:.1} tok/s",
        stats.gen_tokens,
        stats.decode_secs,
        stats.decode_tokens_per_sec()
    );
    Ok(())
}

fn cmd_eval_ppl(
    model_path: &Path,
    tokenizer_path: &Path,
    input: &Path,
    context: usize,
    max_tokens: Option<usize>,
    threads: Option<usize>,
) -> Result<()> {
    set_threads(threads)?;
    let model = load_model(model_path)?;
    let tokenizer = TextTokenizer::from_file(tokenizer_path)?;
    let text = fs::read_to_string(input)?;

    // Plain-text encoding without special tokens; truncate if requested.
    let mut tokens = tokenizer.encode(&text, false)?;
    if let Some(n) = max_tokens {
        tokens.truncate(n);
    }
    if tokens.len() < 3 {
        return Err(minq::MinqError::Model(format!(
            "{}: only {} tokens after encoding/truncation",
            input.display(),
            tokens.len()
        )));
    }

    eprintln!(
        "eval-ppl: {} tokens, context {}, {} window(s)",
        tokens.len(),
        context,
        tokens.len().div_ceil(context)
    );
    let result = minq::ppl::eval_ppl(&model, &tokens, context, |done, total, cum_ppl| {
        eprintln!("  window {done}/{total} done, cumulative ppl = {cum_ppl:.4}");
    })?;

    println!("model:            {}", model_path.display());
    println!("input:            {}", input.display());
    println!("context:          {context}");
    println!("tokens:           {}", result.total_tokens);
    println!("scored positions: {}", result.scored_positions);
    println!("mean nll:         {:.6}", result.mean_nll);
    println!("ppl:              {:.4}", result.ppl);
    Ok(())
}
