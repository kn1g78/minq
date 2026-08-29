# minq

[中文版 README](README_ZH.md)

A from-scratch CPU inference engine for LLaMA-family transformers (LLaMA /
Qwen2 / Qwen3 style), written in Rust **without any existing deep-learning
framework**. Every piece — tensors, block-quantized kernels, the transformer
forward pass, KV-cached decoding, sampling — is implemented by hand on top of
`rayon`, with hand-written AVX2+FMA kernels for the hot paths.

## Architecture

```
                 ┌─────────────────────────── minq ───────────────────────────┐
                 │                                                              │
  prompt ──────► │  tokenizer ──► engine ──────────────────────► detokenized    │
  (text)         │  (BPE wrap)    (prefill + decode loop)        text           │
                 │                   │ 1 fwd pass/prompt  │ 1 fwd pass/token    │
                 │                   ▼                    ▼                     │
                 │                 model  ◄──── KV cache (per layer K/V)        │
                 │   embed → [RMSNorm → QKV(+bias) → RoPE → GQA attention       │
                 │            → O proj → +residual → RMSNorm → SwiGLU           │
                 │            → +residual] × N → RMSNorm → lm_head → logits     │
                 │                   │                    │                     │
                 │           tensor (f32 kernels)   quantize (Q8_0 / Q4_0       │
                 │           rayon matmul/matvec    fused block matvec)         │
                 │                   ▲                    ▲                     │
                 │            format: .minq ◄── quantize CLI (from              │
                 │            header+JSON+tensors   safetensors export)         │
                 └──────────────────────────────────────────────────────────────┘
```

## Modules

| Module          | Responsibility |
|-----------------|----------------|
| `tensor.rs`     | Dense f32 `Tensor` (Vec + shape + strides), rayon-parallel `matmul`/`matvec`, `rmsnorm`, `softmax`, elementwise ops; AVX2+FMA `dot` with runtime dispatch |
| `quantize.rs`   | Research core: Q8_0 / Q4_0 block quantization (32 weights/block), quantize/dequantize, and the fused dequantize-multiply `matvec` hot path (scalar + AVX2 kernels) |
| `model.rs`      | The transformer: RMSNorm, RoPE, MHA/GQA attention with KV cache, SwiGLU MLP, optional Qwen3 per-head q/k-norm; weight loading from safetensors (F32/F16/BF16) and `.minq` |
| `format.rs`     | `.minq` file format: 8-byte magic + config JSON + typed tensor records |
| `tokenizer.rs`  | Wrapper over the `tokenizers` crate (HF `tokenizer.json`) |
| `sampler.rs`    | Greedy / temperature / top-k / top-p sampling, seeded and reproducible |
| `engine.rs`     | Generation loop: prefill + incremental decode with streaming callback |
| `ppl.rs`        | Perplexity evaluation: non-overlapping context windows, log-sum-exp NLL in f64, streaming per-position logits |
| `main.rs`       | CLI: `run`, `quantize`, `bench`, `eval-ppl` |

## Building

A recent stable Rust toolchain (>= 1.85) is enough:

```bash
cargo build --release
```

## Usage

```bash
cargo build --release

# Generate text (f32 straight from HuggingFace, or a .minq file)
./target/release/minq run \
  --model models/qwen2-0.5b \
  --tokenizer models/qwen2-0.5b/tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 64 --temperature 0.7 --top-p 0.9

# Export a quantized model (reads config.json + *.safetensors in the directory)
./target/release/minq quantize \
  --input models/qwen2-0.5b \
  --output models/qwen2-0.5b/qwen2-0.5b-q8_0.minq --dtype q8_0

# Benchmark prefill / decode throughput
./target/release/minq bench --model models/qwen2-0.5b/qwen2-0.5b-q8_0.minq \
  --prompt-tokens 128 --gen-tokens 64 --threads 8

# Perplexity on a text file (the quantization quality metric)
./target/release/minq eval-ppl --model models/qwen2-0.5b/qwen2-0.5b-q8_0.minq \
  --tokenizer models/qwen2-0.5b/tokenizer.json \
  --input wikitext2-test.txt --context 512 --max-tokens 2000
```

`--model` accepts a `.minq` file, a single `.safetensors` file, or a
directory containing `config.json` plus one or more `*.safetensors` shards.
Files exported before the rename (extension `.minfer`, magic `MINFER01`)
have the identical binary layout and are still accepted by the loader;
new exports always use magic `MINQ0001`.

Model weights are **not** part of this repository. Put them under `models/`,
one directory per model, e.g. `models/qwen3-4b-base/` with `config.json`,
`tokenizer.json` and `*.safetensors` downloaded from HuggingFace or
ModelScope, plus the `.minq` files you export from them.

## Quantization formats

Both formats split each weight row into blocks of 32 values with one f32
scale per block (GGML-style), so dequantization is a single multiply per
element and can be fused into the matvec accumulation loop:

- **Q8_0** — 36 bytes/block: `d = max|x| / 127`, `x ≈ q · d`, `q ∈ [-127, 127]` i8.
  3.56× smaller than f32, per-element error ≤ `max|x| / 254`.
- **Q4_0** — 20 bytes/block: `d = max|x| / 8`, `x ≈ (code − 8) · d`, 4-bit
  codes packed two per byte (low nibble = element `2i`, high = `2i+1`).
  6.4× smaller than f32, per-element error ≤ `max|x| / 16`.

Activations always stay in f32 — only weight storage is quantized. This is
weight-only quantization (W4/W8, A32), the regime that matters for
memory-bandwidth-bound token decoding on CPUs.

## SIMD kernels

The decode hot path (`QuantizedTensor::matvec` and the f32 `dot` used by
attention and dense weights) has hand-written `std::arch` AVX2+FMA kernels:

- **Q8_0**: 32 i8 codes are sign-extended to f32 in four 8-lane groups and
  accumulated with FMA; the block scale is applied once per block.
- **Q4_0**: nibbles are unpacked with `and`/`srli`+`and`, interleaved back to
  sequential order with `unpacklo/hi_epi8` (so no shuffling of `x` is
  needed), centered by subtracting 8, then follow the same FMA path.
- **Dispatch** is at runtime via `is_x86_feature_detected!("avx2")` +
  `"fma"` (cached in a `OnceLock`); the scalar kernels remain as the portable
  fallback, and all `unsafe` SIMD code sits behind safe public APIs with
  SAFETY comments. Non-AVX2 machines work unchanged.

Measured on an i5-12450HX (12 threads), Qwen2-0.5B, `minq bench
--prompt-tokens 128 --gen-tokens 64`:

| model | phase   | scalar | AVX2+FMA | speedup |
|-------|---------|--------|----------|---------|
| Q8_0  | prefill | 15.5 tok/s | 22.3 tok/s | 1.44x |
| Q8_0  | decode  | 15.2 tok/s | 20.2 tok/s | 1.33x |
| Q4_0  | prefill | 13.4 tok/s | 23.4 tok/s | 1.75x |
| Q4_0  | decode  | 13.5 tok/s | 21.3 tok/s | 1.58x |

Note how the SIMD build flips the Q4_0/Q8_0 ordering: once compute is no
longer the bottleneck, Q4_0's smaller memory footprint wins — exactly the
bandwidth-bound regime weight-only quantization targets.

## Architecture support

- **LLaMA / Qwen2**: RMSNorm, RoPE, MHA/GQA, SwiGLU; optional Q/K/V biases.
- **Qwen3**: per-head RMSNorm on Q and K (`q_norm` / `k_norm`, head_dim-dim,
  applied before RoPE), explicit `head_dim` decoupled from
  `hidden_size / n_heads`, no QKV bias. Whether q/k-norm is applied is driven
  purely by the presence of the norm tensors, so Qwen2 checkpoints keep
  working unchanged. Tied and untied LM heads are both handled.

## Design decisions and trade-offs

- **No framework, small surface.** The whole engine is ~3.3k lines. Matvec over
  rows parallelized with rayon is the entire "runtime"; there is no operator
  graph, no autograd, no device abstraction.
- **Fused dequant-matvec, not dequant-then-matmul.** Decode is one token at a
  time, i.e. purely memory-bound: the win from quantization comes from reading
  4–6× fewer weight bytes. Materializing f32 weights would give that back, so
  `QuantizedTensor::matvec` accumulates `Σ qᵢ·xᵢ` per block and applies the
  scale once per block.
- **Weight-only quantization.** Quantizing activations (W8A8 etc.) buys
  little at batch size 1 on a CPU and costs real accuracy; skipped on purpose.
- **Embeddings stay f32** (they are indexed, not multiplied), as do the tiny
  RMSNorm gains and Q/K/V biases. Everything else that is 2-D and
  block-aligned gets quantized.
- **Two independent forward paths.** `forward` (KV cache, incremental) and
  `forward_full` (one-shot causal sweep) exist side by side so the test suite
  can prove the cache is exact, not just plausible.
- **Custom `.minq` format instead of GGUF.** A teaching format you can
  hexdump: magic, JSON config, length-prefixed tensor records. No metadata
  zoo, no alignment padding rules.
- **Known limitations (honest list):** no memory-mapping of weights (files
  are read into RAM), no flash-attention (decode is matvec-bound so this
  barely matters), single sequence (no batching), SIMD is
  AVX2-only (no AVX-512 / NEON kernels yet).

## Testing

`cargo test` covers, with randomly generated small models (no downloads):

- Q8_0 roundtrip relative error bounded (< 1%), and Q4_0 strictly worse than
  Q8_0 (monotonicity sanity check)
- fused quantized matvec ≡ dequantize-then-matvec
- RMSNorm against a hand-computed reference
- RoPE: norm preservation and the relative-position property
  `⟨RoPE(q,m), RoPE(k,n)⟩ = f(m − n)`
- **KV cache exactness**: incremental decode logits ≡ one-shot full-forward
  logits (also for prefill+decode splits)
- GQA: `n_kv_heads = 1` ≡ MHA with explicitly replicated K/V heads
- end-to-end determinism of greedy generation on a random 2-layer model
  (dim 64, vocab 128)
- sampler laws: `top_k = 1` ≡ greedy, `temperature = 0` ≡ greedy, seeded
  reproducibility
- `.minq` file roundtrip, bad-magic rejection, and acceptance of the legacy
  `MINFER01` magic
- AVX2 kernels ≡ scalar kernels (relative error < 1e-5)
- Qwen3: KV-cache exactness with per-head q/k-norm, q/k-norm is wired in,
  `head_dim` decoupled from `hidden_size / n_heads`

## Model validation

End-to-end runs against the official Qwen2-0.5B weights (fp16 safetensors,
HuggingFace) on an RTX 4050 6GB laptop (CPU inference, 12 threads):

| dtype | size | prefill | decode | greedy output vs fp16 |
|---|---|---|---|---|
| f32 (from fp16) | 1976 MB | 9.2 tok/s | 6.8 tok/s | — |
| Q8_0 | 947 MB | 15.6 tok/s | 15.7 tok/s (2.3×) | **token-identical** (EN + ZH prompts) |
| Q4_0 | 768 MB | 14.2 tok/s | 13.7 tok/s (2.0×) | diverges: repetition loops, factual drift |

Notable: Q4_0 is *slower* than Q8_0 here — nibble-unpacking overhead eats the
bandwidth win at 12 threads. The handwritten AVX2 kernels (see *SIMD
kernels* above) remove that overhead and flip the ordering in Q4_0's favor.

Scale check with **Qwen3-4B-Base** (ModelScope, same machine; the fp16→f32
baseline does not fit in 16 GB RAM, so quantized tiers are compared
directly):

| dtype | size | prefill | decode | quality (greedy, EN + ZH) |
|---|---|---|---|---|
| Q8_0 | 5644 MB (2.85×) | 3.9 tok/s | 4.1 tok/s | coherent, factually correct |
| Q4_0 | 3827 MB (4.20×) | 5.4 tok/s | **5.2 tok/s** | coherent — no repetition-loop degeneration seen at 0.5B |

Two takeaways: with AVX2 kernels in place, Q4_0 beats Q8_0 on speed at 4B;
and the 4-bit quality loss visible at 0.5B disappears at 4B — smaller models
are more fragile to quantization, i.e. the trade-off is scale-dependent.

Memory-wall check with **Qwen3-8B-Base** (ModelScope, same 16 GB machine):
32.8 GB as f32; Q8_0 packs to 11.0 GB and **fails to load** (out of memory),
while Q4_0 at 7.2 GB runs fine:

| dtype | size | prefill | decode | quality (greedy, EN + ZH) |
|---|---|---|---|---|
| Q8_0 | 11005 MB (2.98×) | — (OOM) | — (OOM) | — |
| Q4_0 | 7221 MB (4.54×) | 4.8 tok/s | **4.6 tok/s** | coherent, factually correct |

For edge deployment, **on engines without mmap, 4-bit is not an
optimization for 8B — it is the entry ticket** (mmap-based engines like
llama.cpp can move this wall by paging weights on demand — which is exactly
why mmap is next on this engine's roadmap). Side observation: Q4_0 decode barely slows from 4B to 8B
(5.2 → 4.6 tok/s) — 8B matvecs have more rows, so 12-thread utilization
improves and effective bandwidth rises from ~20 to ~33 GB/s; per-token time
grows sub-linearly with scale.

Quantitative quality: WikiText-2 test, first 10,000 tokens (context 512,
same protocol for all tiers, built-in `eval-ppl`):

| model | dtype | PPL | vs baseline |
|---|---|---|---|
| Qwen2-0.5B | f32 | 19.15 | — |
| Qwen2-0.5B | Q8_0 | 19.14 | **−0.04% (within noise)** |
| Qwen2-0.5B | Q4_0 | 21.76 | +13.7% |
| Qwen3-4B | Q8_0 | 11.01 | — |
| Qwen3-4B | Q4_0 | 12.01 | +9.1% |

Q8_0 is PPL-indistinguishable from fp16 at 0.5B; Q4_0's relative quality
loss shrinks with scale (+13.7% at 0.5B → +9.1% at 4B); and 4B Q4_0 beats
0.5B fp16 in absolute PPL — within an edge memory budget, a bigger model
with more aggressive quantization wins over a smaller model with high
fidelity, independently reproducing the core finding of the k-bit
inference-scaling-laws literature (Dettmers & Zettlemoyer, 2023).
