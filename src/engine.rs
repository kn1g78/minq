//! The generation loop: prefill the KV cache with the prompt, then decode
//! one token per forward pass, streaming each sampled token to a callback.

use std::time::Instant;

use crate::model::{KVCache, Model};
use crate::sampler::Sampler;
use crate::{MinqError, Result};

/// Timing and token-count statistics for one generation run.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

impl GenStats {
    pub fn prefill_tokens_per_sec(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_secs.max(1e-9)
    }

    pub fn decode_tokens_per_sec(&self) -> f64 {
        self.gen_tokens as f64 / self.decode_secs.max(1e-9)
    }
}

/// Model + sampler + stop conditions, with the prefill/decode loop.
pub struct Engine {
    pub model: Model,
    pub sampler: Sampler,
    pub stop_tokens: Vec<u32>,
}

impl Engine {
    pub fn new(model: Model, sampler: Sampler) -> Self {
        Self {
            model,
            sampler,
            stop_tokens: Vec::new(),
        }
    }

    /// Generate up to `max_tokens` tokens after `prompt`.
    ///
    /// `on_token` is invoked synchronously for every generated token (the
    /// streaming interface used by the CLI); its errors propagate, aborting
    /// generation. Generation stops early if a token in `stop_tokens` is
    /// produced; that stop token is still reported to the callback and
    /// counted.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenStats> {
        if prompt.is_empty() {
            return Err(MinqError::Model("empty prompt".into()));
        }
        let mut cache = KVCache::new(&self.model.config);

        // Prefill: one forward pass over the whole prompt fills the cache.
        let t0 = Instant::now();
        let mut logits = self.model.forward(prompt, &mut cache)?;
        let prefill = t0.elapsed();

        // Decode: one token per step.
        let t1 = Instant::now();
        let mut gen = 0usize;
        for _ in 0..max_tokens {
            let token = self.sampler.sample(&logits);
            on_token(token)?;
            gen += 1;
            if self.stop_tokens.contains(&token) {
                break;
            }
            logits = self.model.forward(&[token], &mut cache)?;
        }
        let decode = t1.elapsed();

        Ok(GenStats {
            prompt_tokens: prompt.len(),
            gen_tokens: gen,
            prefill_secs: prefill.as_secs_f64(),
            decode_secs: decode.as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::tiny_config;
    use crate::model::Model;

    #[test]
    fn greedy_generation_is_deterministic() {
        // End-to-end: a randomly initialized 2-layer model (dim 64, vocab 128)
        // must produce identical token sequences across two identical runs.
        let prompt: Vec<u32> = vec![1, 2, 3];
        let run = || {
            let model = Model::random(tiny_config(4), 1);
            let mut engine = Engine::new(model, Sampler::new(0.0, 0, 1.0, 99));
            let mut out = Vec::new();
            let stats = engine
                .generate(&prompt, 16, |t| {
                    out.push(t);
                    Ok(())
                })
                .expect("generation failed");
            assert_eq!(stats.gen_tokens, 16);
            out
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = Model::random(tiny_config(2), 5);
        let mut engine = Engine::new(model, Sampler::greedy());
        assert!(engine.generate(&[], 4, |_| Ok(())).is_err());
    }

    #[test]
    fn callback_errors_abort_generation() {
        let model = Model::random(tiny_config(2), 6);
        let mut engine = Engine::new(model, Sampler::greedy());
        let mut seen = 0;
        let result = engine.generate(&[1, 2], 8, |_| {
            seen += 1;
            if seen == 2 {
                return Err(MinqError::Model("sink closed".into()));
            }
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(seen, 2);
    }
}
