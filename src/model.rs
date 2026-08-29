//! LLaMA-family transformer, written from scratch.
//!
//! One decoder layer is:
//!
//! ```text
//! x += O( Attention( RoPE(Q rmsnorm(x)), RoPE(K rmsnorm(x)), V rmsnorm(x)) )
//! x += Down( SiLU(Gate(rmsnorm(x))) * Up(rmsnorm(x)) )
//! ```
//!
//! with RMSNorm instead of LayerNorm, rotary position embeddings (RoPE),
//! multi-head / grouped-query attention (MHA/GQA) backed by a KV cache for
//! incremental decoding, and a SwiGLU MLP. Weight matrices may be dense f32
//! (loaded from safetensors) or block-quantized (loaded from `.minq`);
//! [`Linear`] dispatches between the two matvec kernels.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::format::{read_minq, WeightTensor};
use crate::quantize::QuantizedTensor;
use crate::tensor::{self, Tensor};
use crate::{MinqError, Result};

/// Static hyper-parameters of the model, serialized into `.minq` files.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Qwen2-style Q/K/V projections carry biases; LLaMA does not.
    pub has_qkv_bias: bool,
    /// Weight storage tag: "f32", "q8_0" or "q4_0".
    pub dtype: String,
}

/// A weight matrix `[out_features, in_features]` in either dense or
/// block-quantized storage.
#[derive(Clone, Debug, PartialEq)]
pub enum Linear {
    F32(Tensor),
    Quant(QuantizedTensor),
}

impl Linear {
    /// `y = W @ x`.
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        match self {
            Linear::F32(t) => tensor::matvec(t, x),
            Linear::Quant(q) => q.matvec(x),
        }
    }
}

/// Weights of a single decoder layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    /// Qwen3-style per-head RMSNorm on Q/K (head_dim-dim gain), applied
    /// before RoPE. `None` for LLaMA/Qwen2.
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// A fully assembled model.
#[derive(Clone, Debug)]
pub struct Model {
    pub config: ModelConfig,
    /// Token embedding table `[vocab_size, hidden_size]`, always dense f32
    /// (it is indexed, not multiplied).
    pub embed: Tensor,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Linear,
    /// Precomputed RoPE cos/sin, built once at load time.
    pub rope: RopeTables,
}

/// Per-layer key/value history. Each cache stores, per position, a flat
/// `[n_kv_heads * head_dim]` vector; keys are stored post-RoPE.
#[derive(Clone, Debug, Default)]
pub struct LayerCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// KV cache for incremental decoding.
#[derive(Clone, Debug)]
pub struct KVCache {
    pub layers: Vec<LayerCache>,
    /// Number of positions currently cached (same for every layer).
    pub len: usize,
}

impl KVCache {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            layers: vec![LayerCache::default(); config.n_layers],
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.k.clear();
            layer.v.clear();
        }
        self.len = 0;
    }
}

/// Apply rotary position embeddings in place.
///
/// GPT-NeoX style (as used by HF LLaMA/Qwen2): each head vector is split in
/// two halves and element pairs `(i, i + half)` are rotated by `pos * theta^(-2i/d)`.
///
/// Kept for tests and reference; the forward pass uses [`RopeTables`], which
/// precomputes exactly these cos/sin values once at load time.
pub fn apply_rope(x: &mut [f32], pos: usize, n_heads: usize, head_dim: usize, theta: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x1 = head[i];
            let x2 = head[i + half];
            head[i] = x1 * cos - x2 * sin;
            head[i + half] = x2 * cos + x1 * sin;
        }
    }
}

/// Precomputed RoPE tables: `cos`/`sin` of `pos * theta^(-2i/head_dim)` for
/// every position and frequency index, built once at load time. The values
/// are produced by the same f32 expression as [`apply_rope`], so results are
/// bit-identical to computing them on the fly.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTables {
    /// Row-major `[max_seq_len, head_dim/2]`.
    pub cos: Vec<f32>,
    /// Row-major `[max_seq_len, head_dim/2]`.
    pub sin: Vec<f32>,
    pub half: usize,
}

impl RopeTables {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        debug_assert_eq!(head_dim % 2, 0);
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half];
        let mut sin = vec![0.0f32; max_seq_len * half];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos[pos * half + i] = c;
                sin[pos * half + i] = s;
            }
        }
        Self { cos, sin, half }
    }

    /// Rotate every `2 * half`-sized head in `x` at position `pos`.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        debug_assert_eq!(x.len() % (2 * self.half), 0);
        let cos = &self.cos[pos * self.half..(pos + 1) * self.half];
        let sin = &self.sin[pos * self.half..(pos + 1) * self.half];
        for head in x.chunks_exact_mut(2 * self.half) {
            for i in 0..self.half {
                let x1 = head[i];
                let x2 = head[i + self.half];
                head[i] = x1 * cos[i] - x2 * sin[i];
                head[i + self.half] = x2 * cos[i] + x1 * sin[i];
            }
        }
    }
}

/// Qwen3 q/k-norm: RMSNorm applied independently to each head's `head_dim`
/// slice, sharing one `head_dim`-sized gain vector.
fn apply_per_head_norm(x: &mut [f32], weight: &[f32], eps: f32, head_dim: usize) {
    debug_assert_eq!(weight.len(), head_dim);
    debug_assert_eq!(x.len() % head_dim, 0);
    for head in x.chunks_exact_mut(head_dim) {
        let ss = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for (v, &w) in head.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}

/// Single-query attention over a contiguous KV history.
///
/// `q` is `[n_heads * head_dim]`; `k`/`v` are `len` rows of
/// `[n_kv_heads * head_dim]`. Returns `[n_heads * head_dim]`. GQA is handled
/// by mapping query head `h` to KV head `h / (n_heads / n_kv_heads)`.
fn attend(q: &[f32], k: &[f32], v: &[f32], len: usize, cfg: &ModelConfig) -> Vec<f32> {
    let hd = cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * hd;
    let group = cfg.n_heads / cfg.n_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; cfg.n_heads * hd];
    let mut scores = vec![0.0f32; len];
    for h in 0..cfg.n_heads {
        let qh = &q[h * hd..(h + 1) * hd];
        let kv_head = h / group;
        let kv_off = kv_head * hd;
        for (t, score) in scores.iter_mut().enumerate() {
            let kt = &k[t * kv_dim + kv_off..t * kv_dim + kv_off + hd];
            *score = tensor::dot(qh, kt) * scale;
        }
        tensor::softmax(&mut scores);
        let oh = &mut out[h * hd..(h + 1) * hd];
        for (t, &w) in scores.iter().enumerate() {
            let vt = &v[t * kv_dim + kv_off..t * kv_dim + kv_off + hd];
            for (o, &vv) in oh.iter_mut().zip(vt.iter()) {
                *o += w * vv;
            }
        }
    }
    out
}

impl Model {
    /// Run `tokens` through the model, appending to `cache`, and return the
    /// logits of the last token. Used for both prefill (a whole prompt) and
    /// decode (a single token).
    pub fn forward(&self, tokens: &[u32], cache: &mut KVCache) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(MinqError::Model("empty token sequence".into()));
        }
        let mut logits = Vec::new();
        for &tok in tokens {
            if cache.len >= self.config.max_seq_len {
                return Err(MinqError::Model(format!(
                    "sequence exceeds max_seq_len ({})",
                    self.config.max_seq_len
                )));
            }
            logits = self.forward_token(tok, cache.len, cache)?;
            cache.len += 1;
        }
        Ok(logits)
    }

    /// One token at position `pos`, appending K/V to the cache.
    pub fn forward_token(&self, token: u32, pos: usize, cache: &mut KVCache) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let mut x = self.embed.row(token as usize)?.to_vec();
        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention sub-layer ---
            let xn = tensor::rmsnorm(&x, &layer.attn_norm, cfg.rms_norm_eps);
            let mut q = layer.q.matvec(&xn)?;
            let mut k = layer.k.matvec(&xn)?;
            let mut v = layer.v.matvec(&xn)?;
            if let Some(b) = &layer.q_bias {
                tensor::add_(&mut q, b);
            }
            if let Some(b) = &layer.k_bias {
                tensor::add_(&mut k, b);
            }
            if let Some(b) = &layer.v_bias {
                tensor::add_(&mut v, b);
            }
            // Qwen3: per-head q/k RMSNorm before RoPE.
            if let Some(w) = &layer.q_norm {
                apply_per_head_norm(&mut q, w, cfg.rms_norm_eps, cfg.head_dim);
            }
            if let Some(w) = &layer.k_norm {
                apply_per_head_norm(&mut k, w, cfg.rms_norm_eps, cfg.head_dim);
            }
            self.rope.apply(&mut q, pos);
            self.rope.apply(&mut k, pos);
            cache.layers[li].k.extend_from_slice(&k);
            cache.layers[li].v.extend_from_slice(&v);
            let lc = &cache.layers[li];
            let attn = attend(&q, &lc.k, &lc.v, pos + 1, cfg);
            let proj = layer.o.matvec(&attn)?;
            tensor::add_(&mut x, &proj);

            // --- SwiGLU MLP sub-layer ---
            let xn = tensor::rmsnorm(&x, &layer.ffn_norm, cfg.rms_norm_eps);
            let gate = layer.gate.matvec(&xn)?;
            let up = layer.up.matvec(&xn)?;
            let mut hidden = tensor::silu(&gate);
            tensor::mul_(&mut hidden, &up);
            let down = layer.down.matvec(&hidden)?;
            tensor::add_(&mut x, &down);
        }
        let xn = tensor::rmsnorm(&x, &self.final_norm, cfg.rms_norm_eps);
        self.lm_head.matvec(&xn)
    }

    /// Reference one-shot forward pass: every position is processed in a
    /// single causal sweep with no cache. Returns the last token's logits.
    ///
    /// Kept as an independent code path from [`Model::forward`] so tests can
    /// cross-validate the KV-cache implementation against it.
    pub fn forward_full(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let xs = self.forward_full_states(tokens)?;
        let cfg = &self.config;
        let xn = tensor::rmsnorm(&xs[xs.len() - 1], &self.final_norm, cfg.rms_norm_eps);
        self.lm_head.matvec(&xn)
    }

    /// One-shot causal forward pass returning the per-position hidden states
    /// after the last decoder layer (before the final RMSNorm and LM head).
    ///
    /// Used by perplexity evaluation, which needs every position's logits;
    /// [`Model::forward_full`] is a thin wrapper over this.
    pub fn forward_full_states(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(MinqError::Model("empty token sequence".into()));
        }
        let cfg = &self.config;
        let n = tokens.len();
        if n > cfg.max_seq_len {
            return Err(MinqError::Model(format!(
                "sequence exceeds max_seq_len ({})",
                cfg.max_seq_len
            )));
        }
        let hd = cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * hd;
        let mut xs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| self.embed.row(t as usize).map(|r| r.to_vec()))
            .collect::<Result<_>>()?;

        for layer in &self.layers {
            // Project Q/K/V for every position, then apply RoPE.
            let mut qs = Vec::with_capacity(n);
            let mut ks = Vec::with_capacity(n);
            let mut vs = Vec::with_capacity(n);
            for x in xs.iter() {
                let xn = tensor::rmsnorm(x, &layer.attn_norm, cfg.rms_norm_eps);
                let mut q = layer.q.matvec(&xn)?;
                let mut k = layer.k.matvec(&xn)?;
                let v = layer.v.matvec(&xn)?;
                if let Some(b) = &layer.q_bias {
                    tensor::add_(&mut q, b);
                }
                if let Some(b) = &layer.k_bias {
                    tensor::add_(&mut k, b);
                }
                qs.push(q);
                ks.push(k);
                vs.push(match &layer.v_bias {
                    Some(b) => {
                        let mut v = v;
                        tensor::add_(&mut v, b);
                        v
                    }
                    None => v,
                });
            }
            for (pos, q) in qs.iter_mut().enumerate() {
                if let Some(w) = &layer.q_norm {
                    apply_per_head_norm(q, w, cfg.rms_norm_eps, hd);
                }
                self.rope.apply(q, pos);
            }
            for (pos, k) in ks.iter_mut().enumerate() {
                if let Some(w) = &layer.k_norm {
                    apply_per_head_norm(k, w, cfg.rms_norm_eps, hd);
                }
                self.rope.apply(k, pos);
            }
            let k_flat: Vec<f32> = ks.concat();
            let v_flat: Vec<f32> = vs.concat();

            // Causal attention: position t attends to 0..=t.
            for (t, x) in xs.iter_mut().enumerate() {
                let attn = attend(
                    &qs[t],
                    &k_flat[..(t + 1) * kv_dim],
                    &v_flat[..(t + 1) * kv_dim],
                    t + 1,
                    cfg,
                );
                let proj = layer.o.matvec(&attn)?;
                tensor::add_(x, &proj);
            }

            // SwiGLU MLP.
            for x in xs.iter_mut() {
                let xn = tensor::rmsnorm(x, &layer.ffn_norm, cfg.rms_norm_eps);
                let gate = layer.gate.matvec(&xn)?;
                let up = layer.up.matvec(&xn)?;
                let mut hidden = tensor::silu(&gate);
                tensor::mul_(&mut hidden, &up);
                let down = layer.down.matvec(&hidden)?;
                tensor::add_(x, &down);
            }
        }

        Ok(xs)
    }

    /// Deterministically initialized random model for tests and smoke runs.
    /// Weights are uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`; norm gains
    /// are 1.0 and biases 0.
    pub fn random(config: ModelConfig, seed: u64) -> Model {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = |rows: usize, cols: usize| {
            let scale = (cols as f32).sqrt().recip();
            let data: Vec<f32> = (0..rows * cols)
                .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            Tensor::new(data, vec![rows, cols]).unwrap()
        };
        let cfg = &config;
        let d = cfg.hidden_size;
        let qd = cfg.n_heads * cfg.head_dim;
        let kvd = cfg.n_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let embed = matrix(cfg.vocab_size, d);
        let lm_head = Linear::F32(matrix(cfg.vocab_size, d));
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: vec![1.0; d],
                q: Linear::F32(matrix(qd, d)),
                k: Linear::F32(matrix(kvd, d)),
                v: Linear::F32(matrix(kvd, d)),
                o: Linear::F32(matrix(d, qd)),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
                ffn_norm: vec![1.0; d],
                gate: Linear::F32(matrix(inter, d)),
                up: Linear::F32(matrix(inter, d)),
                down: Linear::F32(matrix(d, inter)),
            });
        }
        Model {
            rope: RopeTables::new(config.head_dim, config.max_seq_len, config.rope_theta),
            config: config.clone(),
            embed,
            layers,
            final_norm: vec![1.0; d],
            lm_head,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

type TensorGetter<'a> = dyn FnMut(&str) -> Result<Option<WeightTensor>> + 'a;

fn need(get: &mut TensorGetter, name: &str) -> Result<WeightTensor> {
    get(name)?.ok_or_else(|| MinqError::Model(format!("missing tensor `{name}`")))
}

fn get_linear(get: &mut TensorGetter, name: &str) -> Result<Linear> {
    Ok(match need(get, name)? {
        WeightTensor::F32(t) => {
            if t.ndim() != 2 {
                return Err(MinqError::Model(format!(
                    "tensor `{name}` must be 2-D, got {:?}",
                    t.shape
                )));
            }
            Linear::F32(t)
        }
        WeightTensor::Quant(q) => Linear::Quant(q),
    })
}

fn get_vec(get: &mut TensorGetter, name: &str) -> Result<Vec<f32>> {
    match need(get, name)? {
        WeightTensor::F32(t) if t.ndim() == 1 => Ok(t.data),
        other => Err(MinqError::Model(format!(
            "tensor `{name}` must be a 1-D f32 vector, got {other:?}"
        ))),
    }
}

fn get_opt_vec(get: &mut TensorGetter, name: &str) -> Result<Option<Vec<f32>>> {
    match get(name)? {
        None => Ok(None),
        Some(WeightTensor::F32(t)) if t.ndim() == 1 => Ok(Some(t.data)),
        Some(other) => Err(MinqError::Model(format!(
            "tensor `{name}` must be a 1-D f32 vector, got {other:?}"
        ))),
    }
}

/// Assemble a [`Model`] from named tensors, validating the basic geometry.
fn assemble(config: &ModelConfig, get: &mut TensorGetter) -> Result<Model> {
    if config.head_dim % 2 != 0 {
        return Err(MinqError::Model("head_dim must be even (RoPE)".into()));
    }
    if config.n_heads % config.n_kv_heads != 0 {
        return Err(MinqError::Model(
            "n_heads must be a multiple of n_kv_heads".into(),
        ));
    }
    let embed = match need(get, "model.embed_tokens.weight")? {
        WeightTensor::F32(t) if t.ndim() == 2 => t,
        other => {
            return Err(MinqError::Model(format!(
                "model.embed_tokens.weight must be 2-D f32, got {other:?}"
            )))
        }
    };
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let p = format!("model.layers.{i}");
        layers.push(LayerWeights {
            attn_norm: get_vec(get, &format!("{p}.input_layernorm.weight"))?,
            q: get_linear(get, &format!("{p}.self_attn.q_proj.weight"))?,
            k: get_linear(get, &format!("{p}.self_attn.k_proj.weight"))?,
            v: get_linear(get, &format!("{p}.self_attn.v_proj.weight"))?,
            o: get_linear(get, &format!("{p}.self_attn.o_proj.weight"))?,
            q_bias: get_opt_vec(get, &format!("{p}.self_attn.q_proj.bias"))?,
            k_bias: get_opt_vec(get, &format!("{p}.self_attn.k_proj.bias"))?,
            v_bias: get_opt_vec(get, &format!("{p}.self_attn.v_proj.bias"))?,
            // Present in Qwen3, absent in LLaMA/Qwen2 — presence drives use.
            q_norm: get_opt_vec(get, &format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: get_opt_vec(get, &format!("{p}.self_attn.k_norm.weight"))?,
            ffn_norm: get_vec(get, &format!("{p}.post_attention_layernorm.weight"))?,
            gate: get_linear(get, &format!("{p}.mlp.gate_proj.weight"))?,
            up: get_linear(get, &format!("{p}.mlp.up_proj.weight"))?,
            down: get_linear(get, &format!("{p}.mlp.down_proj.weight"))?,
        });
    }
    let final_norm = get_vec(get, "model.norm.weight")?;
    // Tied embeddings: fall back to the embedding table as the LM head.
    let lm_head = match get("lm_head.weight")? {
        Some(_) => get_linear(get, "lm_head.weight")?,
        None => Linear::F32(embed.clone()),
    };
    Ok(Model {
        config: config.clone(),
        embed,
        layers,
        final_norm,
        lm_head,
        rope: RopeTables::new(config.head_dim, config.max_seq_len, config.rope_theta),
    })
}

/// Resolve a model path to `(config directory, safetensors files)`.
/// Accepts a directory (all `*.safetensors` inside, sorted) or a single file.
pub fn resolve_safetensors(path: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    let (dir, files) = if path.is_dir() {
        let mut files: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        files.sort();
        (path.to_path_buf(), files)
    } else {
        let dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        (dir, vec![path.to_path_buf()])
    };
    if files.is_empty() {
        return Err(MinqError::Model(format!(
            "{}: no .safetensors files found",
            path.display()
        )));
    }
    Ok((dir, files))
}

/// Parse a HuggingFace `config.json` (LLaMA / Qwen2 field names).
pub fn hf_config(dir: &Path) -> Result<ModelConfig> {
    let text = fs::read_to_string(dir.join("config.json"))?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let get_u = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| MinqError::Model(format!("config.json missing `{key}`")))
    };
    let get_f = |key: &str| v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32);
    let hidden = get_u("hidden_size")?;
    let heads = get_u("num_attention_heads")?;
    Ok(ModelConfig {
        hidden_size: hidden,
        intermediate_size: get_u("intermediate_size")?,
        n_layers: get_u("num_hidden_layers")?,
        n_heads: heads,
        n_kv_heads: get_u("num_key_value_heads").unwrap_or(heads),
        head_dim: get_u("head_dim").unwrap_or(hidden / heads),
        vocab_size: get_u("vocab_size")?,
        max_seq_len: get_u("max_position_embeddings").unwrap_or(4096),
        rope_theta: get_f("rope_theta").unwrap_or(10000.0),
        rms_norm_eps: get_f("rms_norm_eps").unwrap_or(1e-6),
        has_qkv_bias: false, // refined by the caller from tensor presence
        dtype: "f32".to_string(),
    })
}

/// Read only the safetensors header (8-byte length + JSON) and return the
/// tensor names, without touching any weight data. Used by the quantize
/// command to plan a streaming export.
pub fn scan_safetensors_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read as _;
    let mut f = fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    // Header JSON is tiny in practice; cap it to reject corrupt files before
    // allocating.
    if header_len > 64 * 1024 * 1024 {
        return Err(MinqError::Model(format!(
            "{}: safetensors header of {header_len} bytes looks corrupt",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    f.read_exact(&mut header)?;
    let v: serde_json::Value = serde_json::from_slice(&header)?;
    let obj = v
        .as_object()
        .ok_or_else(|| MinqError::Model(format!("{}: bad safetensors header", path.display())))?;
    Ok(obj
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect())
}

/// Convert one safetensors tensor (F32 / F16 / BF16) to a dense f32 tensor.
pub fn st_to_tensor(st: &SafeTensors, name: &str) -> Result<Tensor> {
    let view = st.tensor(name)?;
    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<f32> = match view.dtype() {
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => {
            return Err(MinqError::Model(format!(
                "tensor `{name}`: unsupported dtype {other:?} (need F32/F16/BF16)"
            )))
        }
    };
    Tensor::new(data, shape)
}

/// Load a dense f32 model from safetensors weights (HF LLaMA/Qwen2 naming).
pub fn load_safetensors(path: &Path) -> Result<Model> {
    let (dir, files) = resolve_safetensors(path)?;
    let mut config = hf_config(&dir)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for file in &files {
        let buf = fs::read(file)?;
        let st = SafeTensors::deserialize(&buf)?;
        for name in st.names() {
            tensors.insert(name.to_string(), st_to_tensor(&st, name)?);
        }
    }
    config.has_qkv_bias = tensors.contains_key("model.layers.0.self_attn.q_proj.bias");
    config.dtype = "f32".to_string();
    assemble(&config, &mut |name| {
        Ok(tensors.get(name).cloned().map(WeightTensor::F32))
    })
}

/// Load a model from a `.minq` file (dense or quantized).
pub fn load_minq(path: &Path) -> Result<Model> {
    let (config, records) = read_minq(path)?;
    let map: HashMap<String, WeightTensor> = records.into_iter().collect();
    assemble(&config, &mut |name| Ok(map.get(name).cloned()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tensor::dot;

    /// Tiny 2-layer config (dim 64, vocab 128) used across the test suite.
    pub(crate) fn tiny_config(n_kv_heads: usize) -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            intermediate_size: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads,
            head_dim: 16,
            vocab_size: 128,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            has_qkv_bias: false,
            dtype: "f32".to_string(),
        }
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (rng.gen::<f32>() - 0.5) * 2.0).collect()
    }

    #[test]
    fn rope_preserves_vector_norm() {
        // Rotation is orthogonal: per-pair norm, hence total norm, is kept.
        let n_heads = 2;
        let head_dim = 16;
        let mut x = rand_vec(n_heads * head_dim, 1);
        let before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        apply_rope(&mut x, 37, n_heads, head_dim, 10000.0);
        let after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((before - after).abs() < 1e-4, "{before} vs {after}");
    }

    #[test]
    fn rope_tables_match_on_the_fly_computation() {
        // The precomputed tables must be bit-identical to apply_rope's
        // per-call math (same f32 expression, evaluated once up front).
        let (head_dim, max_seq, theta) = (16usize, 64usize, 10000.0f32);
        let tables = RopeTables::new(head_dim, max_seq, theta);
        for pos in [0usize, 1, 7, 33, 63] {
            let mut a = rand_vec(2 * head_dim, 100 + pos as u64);
            let mut b = a.clone();
            apply_rope(&mut a, pos, 2, head_dim, theta);
            tables.apply(&mut b, pos);
            assert_eq!(a, b, "pos {pos} differs");
        }
    }

    #[test]
    fn rope_encodes_relative_positions() {
        // dot(RoPE(q, m), RoPE(k, n)) must depend only on m - n.
        let head_dim = 16;
        let q = rand_vec(head_dim, 2);
        let k = rand_vec(head_dim, 3);
        let (m, n, delta) = (9usize, 3usize, 5usize);

        let mut q1 = q.clone();
        let mut k1 = k.clone();
        apply_rope(&mut q1, m, 1, head_dim, 10000.0);
        apply_rope(&mut k1, n, 1, head_dim, 10000.0);

        let mut q2 = q.clone();
        let mut k2 = k.clone();
        apply_rope(&mut q2, m + delta, 1, head_dim, 10000.0);
        apply_rope(&mut k2, n + delta, 1, head_dim, 10000.0);

        let d1 = dot(&q1, &k1);
        let d2 = dot(&q2, &k2);
        assert!((d1 - d2).abs() < 1e-4, "{d1} vs {d2}");
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        let model = Model::random(tiny_config(4), 7);
        let tokens: Vec<u32> = vec![5, 11, 23, 97, 3, 42];
        let reference = model.forward_full(&tokens).unwrap();

        // (a) pure incremental decoding, one token at a time
        let mut cache = KVCache::new(&model.config);
        let mut last = Vec::new();
        for &t in &tokens {
            last = model.forward(&[t], &mut cache).unwrap();
        }
        assert_logits_close(&reference, &last);

        // (b) prefill + decode split must give the same result
        let mut cache = KVCache::new(&model.config);
        model.forward(&tokens[..4], &mut cache).unwrap();
        let last = model.forward(&tokens[4..], &mut cache).unwrap();
        assert_logits_close(&reference, &last);
    }

    #[test]
    fn gqa_broadcast_matches_expanded_kv() {
        // A GQA model with n_kv_heads = 1 must behave exactly like an MHA
        // model whose K/V projections are copies of the single KV head.
        let mut gqa = Model::random(tiny_config(1), 42);
        let mut mha = gqa.clone();
        mha.config.n_kv_heads = 4;
        for (gqa_layer, mha_layer) in gqa.layers.iter_mut().zip(mha.layers.iter_mut()) {
            // Keep every weight identical except K/V, which we expand 4x.
            for (src, dst) in [
                (&gqa_layer.k, &mut mha_layer.k),
                (&gqa_layer.v, &mut mha_layer.v),
            ] {
                if let (Linear::F32(t), Linear::F32(d)) = (src, dst) {
                    let (rows, cols) = (t.shape[0], t.shape[1]);
                    let mut data = Vec::with_capacity(4 * rows * cols);
                    for _ in 0..4 {
                        data.extend_from_slice(&t.data);
                    }
                    *d = Tensor::new(data, vec![4 * rows, cols]).unwrap();
                }
            }
        }
        gqa.config.n_kv_heads = 1;
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let a = gqa.forward_full(&tokens).unwrap();
        let b = mha.forward_full(&tokens).unwrap();
        assert_logits_close(&a, &b);
    }

    /// Qwen3-style config: explicit head_dim decoupled from
    /// hidden_size / n_heads (64 / 4 = 16, but head_dim = 8), no QKV bias.
    fn qwen3_tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            intermediate_size: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            vocab_size: 128,
            max_seq_len: 64,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            has_qkv_bias: false,
            dtype: "f32".to_string(),
        }
    }

    /// Random Qwen3-style model: adds per-head q/k RMSNorm gains.
    fn qwen3_random(seed: u64) -> Model {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut model = Model::random(qwen3_tiny_config(), seed);
        let mut rng = StdRng::seed_from_u64(seed + 1);
        let hd = model.config.head_dim;
        for layer in &mut model.layers {
            layer.q_norm = Some((0..hd).map(|_| 0.5 + rng.gen::<f32>()).collect());
            layer.k_norm = Some((0..hd).map(|_| 0.5 + rng.gen::<f32>()).collect());
        }
        model
    }

    #[test]
    fn qwen3_kv_cache_matches_full_forward() {
        let model = qwen3_random(17);
        let tokens: Vec<u32> = vec![5, 11, 23, 97, 3, 42];
        let reference = model.forward_full(&tokens).unwrap();

        // Pure incremental decoding.
        let mut cache = KVCache::new(&model.config);
        let mut last = Vec::new();
        for &t in &tokens {
            last = model.forward(&[t], &mut cache).unwrap();
        }
        assert_logits_close(&reference, &last);

        // Prefill + decode split.
        let mut cache = KVCache::new(&model.config);
        model.forward(&tokens[..2], &mut cache).unwrap();
        let last = model.forward(&tokens[2..], &mut cache).unwrap();
        assert_logits_close(&reference, &last);
    }

    #[test]
    fn qk_norm_actually_changes_output() {
        // Sanity: the q/k norms must be wired in, not silently ignored.
        let with_norms = qwen3_random(19);
        let mut without = with_norms.clone();
        for layer in &mut without.layers {
            layer.q_norm = None;
            layer.k_norm = None;
        }
        let tokens: Vec<u32> = vec![1, 2, 3];
        let a = with_norms.forward_full(&tokens).unwrap();
        let b = without.forward_full(&tokens).unwrap();
        let max_diff: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(max_diff > 1e-3, "q/k norms had no effect ({max_diff})");
    }

    #[test]
    fn head_dim_decoupled_from_hidden_size() {
        // qwen3_tiny_config: hidden/n_heads = 16 but head_dim = 8, so the Q
        // projection is [32, 64] and O is [64, 32]. This fails loudly if any
        // code path assumes head_dim == hidden_size / n_heads.
        let model = qwen3_random(23);
        let logits = model.forward_full(&[7, 8, 9]).unwrap();
        assert_eq!(logits.len(), model.config.vocab_size);
    }

    fn assert_logits_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        let max_diff: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 1e-3, "logits differ by up to {max_diff}");
    }
}
