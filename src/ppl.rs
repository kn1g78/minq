//! Perplexity evaluation — the quality metric of the quantization
//! quality–speed trade-off study.
//!
//! Method: the input token sequence is split into non-overlapping windows of
//! `context` tokens; each window is scored with a single one-shot forward
//! pass ([`Model::forward_full_states`]). Position `t` predicts token
//! `t + 1`; position 0 of every window is skipped (no left context), and the
//! last position of a window predicts nothing, so a window of length L
//! contributes `max(L - 2, 0)` scored positions. All models are measured
//! under this same convention so numbers stay comparable.

use crate::model::Model;
use crate::tensor;
use crate::{MinqError, Result};

/// Aggregate result of a perplexity run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PplResult {
    /// Tokens in the (possibly truncated) input sequence.
    pub total_tokens: usize,
    /// Positions that contributed to the average.
    pub scored_positions: usize,
    /// Mean cross-entropy (natural log) over scored positions.
    pub mean_nll: f64,
    /// `exp(mean_nll)`.
    pub ppl: f64,
}

/// Negative log-likelihood of `target` under `logits`, via log-sum-exp.
fn nll(logits: &[f32], target: u32) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let logsumexp = max as f64 + sum_exp.ln();
    logsumexp - logits[target as usize] as f64
}

/// Evaluate perplexity of `tokens` under `model`.
///
/// `on_window(done, total, cumulative_ppl)` is invoked after each window for
/// progress reporting. Logits are produced one position at a time and
/// accumulated immediately in f64 — with a 150k vocabulary a full
/// `[context, vocab]` logits matrix would be hundreds of MB, so nothing
/// larger than a single logits vector is ever materialized.
pub fn eval_ppl(
    model: &Model,
    tokens: &[u32],
    context: usize,
    mut on_window: impl FnMut(usize, usize, f64),
) -> Result<PplResult> {
    if tokens.len() < 3 {
        return Err(MinqError::Model(format!(
            "need at least 3 tokens to score, got {}",
            tokens.len()
        )));
    }
    if context < 3 {
        return Err(MinqError::Model(format!(
            "context must be >= 3, got {context}"
        )));
    }
    if context > model.config.max_seq_len {
        return Err(MinqError::Model(format!(
            "context {context} exceeds max_seq_len ({})",
            model.config.max_seq_len
        )));
    }

    let n_windows = tokens.len().div_ceil(context);
    let mut nll_sum = 0.0f64;
    let mut count = 0usize;

    for (wi, window) in tokens.chunks(context).enumerate() {
        let states = model.forward_full_states(window)?;
        let eps = model.config.rms_norm_eps;
        // Skip position 0 (no left context) and the last position (its
        // target lies outside the window).
        for t in 1..window.len().saturating_sub(1) {
            let xn = tensor::rmsnorm(&states[t], &model.final_norm, eps);
            let logits = model.lm_head.matvec(&xn)?;
            nll_sum += nll(&logits, window[t + 1]);
            count += 1;
        }
        let cumulative = if count > 0 {
            (nll_sum / count as f64).exp()
        } else {
            f64::NAN
        };
        on_window(wi + 1, n_windows, cumulative);
    }

    if count == 0 {
        return Err(MinqError::Model(
            "no scorable positions (sequence too short for context)".into(),
        ));
    }
    let mean_nll = nll_sum / count as f64;
    Ok(PplResult {
        total_tokens: tokens.len(),
        scored_positions: count,
        mean_nll,
        ppl: mean_nll.exp(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::tiny_config;
    use crate::model::Model;

    /// Deterministic pseudo-random token ids in the tiny vocab (128).
    fn token_seq(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 37 + 11) % 128) as u32).collect()
    }

    #[test]
    fn ppl_is_finite_and_at_least_one() {
        let model = Model::random(tiny_config(4), 5);
        let tokens = token_seq(40);
        let r = eval_ppl(&model, &tokens, 16, |_, _, _| {}).unwrap();
        // Windows are 16/16/8 tokens -> 14 + 14 + 6 scored positions.
        assert_eq!(r.total_tokens, 40);
        assert_eq!(r.scored_positions, 34);
        assert!(r.ppl.is_finite(), "ppl must be finite, got {}", r.ppl);
        assert!(r.ppl >= 1.0, "ppl must be >= 1, got {}", r.ppl);
        assert!((r.ppl - r.mean_nll.exp()).abs() < 1e-9);
    }

    #[test]
    fn ppl_is_similar_across_context_sizes() {
        // Same text, same convention: window size must not change the order
        // of magnitude of the result (positions scored differ slightly at
        // window boundaries, so exact equality is not expected).
        let model = Model::random(tiny_config(4), 6);
        let tokens = token_seq(96);
        let p16 = eval_ppl(&model, &tokens, 16, |_, _, _| {}).unwrap().ppl;
        let p32 = eval_ppl(&model, &tokens, 32, |_, _, _| {}).unwrap().ppl;
        let p64 = eval_ppl(&model, &tokens, 64, |_, _, _| {}).unwrap().ppl;
        for (a, b) in [(p16, p32), (p32, p64), (p16, p64)] {
            let ratio = a / b;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "ppl ratio {ratio} out of range ({a} vs {b})"
            );
        }
    }

    #[test]
    fn ppl_handles_uneven_final_window() {
        // 33 tokens with context 16 -> windows of 16, 16 and 1 token; the
        // 1-token tail has no scorable position and must not panic.
        let model = Model::random(tiny_config(4), 7);
        let tokens = token_seq(33);
        let r = eval_ppl(&model, &tokens, 16, |_, _, _| {}).unwrap();
        assert_eq!(r.scored_positions, 14 + 14);
        assert!(r.ppl.is_finite() && r.ppl >= 1.0);
    }
}
