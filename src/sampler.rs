//! Token sampling: greedy, temperature, top-k and top-p (nucleus).
//!
//! The sampler is seeded so runs are reproducible — important both for the
//! test suite and for benchmarking.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Index of the largest logit; ties resolve to the lowest index.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Configuration + RNG state for next-token sampling.
#[derive(Clone, Debug)]
pub struct Sampler {
    /// `<= 0` means greedy (argmax).
    pub temperature: f32,
    /// Keep only the k highest-probability tokens; `0` disables the filter.
    pub top_k: usize,
    /// Nucleus threshold in `(0, 1]`; `1.0` disables the filter.
    pub top_p: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Greedy sampler (temperature 0).
    pub fn greedy() -> Self {
        Self::new(0.0, 0, 1.0, 0)
    }

    /// Pick the next token from a logits vector.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.temperature <= 0.0 || self.top_k == 1 {
            return argmax(logits);
        }

        // 1. temperature scaling + softmax
        let mut probs: Vec<f32> = logits.iter().map(|l| l / self.temperature).collect();
        crate::tensor::softmax(&mut probs);

        // 2. fast path: no filters — draw directly, no selection or sorting.
        if (self.top_k == 0 || self.top_k >= probs.len()) && self.top_p >= 1.0 {
            let mut r = self.rng.gen::<f32>();
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return i as u32;
                }
            }
            return (probs.len() - 1) as u32;
        }

        // 3. top-k by partial selection: partition the k largest probabilities
        // instead of sorting the whole vocabulary (150k entries for Qwen).
        let k = if self.top_k == 0 {
            probs.len()
        } else {
            self.top_k.min(probs.len())
        };
        let mut ranked: Vec<usize> = (0..probs.len()).collect();
        if k < ranked.len() {
            ranked.select_nth_unstable_by(k - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
            ranked.truncate(k);
        }
        // Only the k survivors need ordering, for the top-p cumulative cutoff.
        ranked.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));

        // 4. top-p (nucleus) cutoff within the top-k set
        let mut cutoff = k;
        let mut cumulative = 0.0f32;
        for (rank, &i) in ranked.iter().enumerate() {
            cumulative += probs[i];
            if cumulative >= self.top_p {
                cutoff = rank + 1;
                break;
            }
        }
        let candidates = &ranked[..cutoff];

        // 5. renormalize and draw
        let total: f32 = candidates.iter().map(|&i| probs[i]).sum();
        let mut r = self.rng.gen::<f32>() * total;
        for &i in candidates {
            r -= probs[i];
            if r <= 0.0 {
                return i as u32;
            }
        }
        // Floating-point fallout guard: return the last (least likely) candidate.
        candidates[candidates.len() - 1] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_logits(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (rng.gen::<f32>() - 0.5) * 20.0).collect()
    }

    #[test]
    fn top_k_1_equals_greedy() {
        let logits = rand_logits(128, 1);
        let mut s = Sampler::new(1.0, 1, 1.0, 123);
        assert_eq!(s.sample(&logits), argmax(&logits));
    }

    #[test]
    fn temperature_zero_equals_greedy() {
        let logits = rand_logits(128, 2);
        let mut s = Sampler::new(0.0, 0, 1.0, 123);
        assert_eq!(s.sample(&logits), argmax(&logits));
    }

    #[test]
    fn sampling_is_reproducible_for_same_seed() {
        let logits = rand_logits(256, 3);
        let mut a = Sampler::new(0.8, 40, 0.9, 42);
        let mut b = Sampler::new(0.8, 40, 0.9, 42);
        let seq_a: Vec<u32> = (0..32).map(|_| a.sample(&logits)).collect();
        let seq_b: Vec<u32> = (0..32).map(|_| b.sample(&logits)).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn samples_are_within_vocabulary() {
        let logits = rand_logits(50, 4);
        let mut s = Sampler::new(1.0, 0, 1.0, 7);
        for _ in 0..64 {
            assert!((s.sample(&logits) as usize) < logits.len());
        }
    }

    #[test]
    fn top_p_keeps_at_least_one_candidate() {
        // Degenerate logits: one token dominates; top-p must still return it.
        let mut logits = vec![-100.0f32; 32];
        logits[5] = 100.0;
        let mut s = Sampler::new(1.0, 0, 0.5, 9);
        assert_eq!(s.sample(&logits), 5);
    }
}
