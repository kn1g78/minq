//! Minimal dense `f32` tensor plus the math kernels used by the forward pass.
//!
//! The tensor type is deliberately small: a contiguous row-major buffer with
//! shape and strides. Everything here is plain Rust; `rayon` provides
//! thread-level parallelism for the row-parallel matrix kernels.

use rayon::prelude::*;

use crate::{MinqError, Result};

/// Contiguous row-major dense tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

/// Row-major strides for a shape: strides[i] = product of shape[i+1..].
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl Tensor {
    /// Build a tensor, checking that the shape matches the data length.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(MinqError::Shape(format!(
                "shape {shape:?} implies {numel} elements, got {}",
                data.len()
            )));
        }
        let strides = compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Zero-filled tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: vec![0.0; numel],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Row `i` of a 2-D matrix as a slice.
    pub fn row(&self, i: usize) -> Result<&[f32]> {
        if self.ndim() != 2 {
            return Err(MinqError::Shape(format!(
                "row() requires a 2-D tensor, got {:?}",
                self.shape
            )));
        }
        let cols = self.shape[1];
        if i >= self.shape[0] {
            return Err(MinqError::Shape(format!(
                "row index {i} out of bounds for {} rows",
                self.shape[0]
            )));
        }
        Ok(&self.data[i * cols..(i + 1) * cols])
    }
}

/// Runtime CPU feature check (cached): AVX2 and FMA are both required by
/// the SIMD kernels. Always `false` on non-x86_64 targets.
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Dot product of two equal-length slices, dispatching to AVX2 when available.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2+FMA availability was just checked; the kernel reads
        // exactly min(a.len(), b.len()) elements, which are equal in length.
        return unsafe { dot_avx2(a, b) };
    }
    dot_scalar(a, b)
}

/// Portable scalar dot product (fallback path).
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2+FMA dot product with two 8-lane accumulators.
///
/// # Safety
///
/// The caller must guarantee AVX2 and FMA are available (checked via
/// [`has_avx2_fma`]). Reads are unaligned loads within `a`/`b`; the tail
/// beyond the last full 16 elements is handled scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s);
    let sums = _mm_add_ps(s, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    for j in i..n {
        total += a[j] * b[j];
    }
    total
}

/// 2-D matrix multiplication: `[m, k] x [k, n] -> [m, n]`, parallel over rows.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 || b.ndim() != 2 || a.shape[1] != b.shape[0] {
        return Err(MinqError::Shape(format!(
            "matmul: incompatible shapes {:?} x {:?}",
            a.shape, b.shape
        )));
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let mut out = vec![0.0f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let arow = &a.data[i * k..(i + 1) * k];
        for (j, cell) in row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (p, &av) in arow.iter().enumerate() {
                sum += av * b.data[p * n + j];
            }
            *cell = sum;
        }
    });
    Tensor::new(out, vec![m, n])
}

/// Matrix-vector product: `w` is `[out_features, in_features]`, result is
/// `w @ x`. This is the decode-time hot path for f32 weights.
pub fn matvec(w: &Tensor, x: &[f32]) -> Result<Vec<f32>> {
    if w.ndim() != 2 || w.shape[1] != x.len() {
        return Err(MinqError::Shape(format!(
            "matvec: weight shape {:?} vs input len {}",
            w.shape,
            x.len()
        )));
    }
    let cols = w.shape[1];
    let out: Vec<f32> = w.data.par_chunks_exact(cols).map(|row| dot(row, x)).collect();
    Ok(out)
}

/// `dst += src` elementwise.
pub fn add_(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// `dst *= src` elementwise.
pub fn mul_(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d *= s;
    }
}

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v / (1.0 + (-v).exp()))
        .collect()
}

/// Root-mean-square normalization with a per-channel gain:
/// `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), weight.len());
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v * inv * w)
        .collect()
}

/// Numerically stable in-place softmax over a slice.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small_known_result() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn rmsnorm_matches_hand_computed_reference() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let eps = 1e-5f32;
        // Reference computed by hand: rms = sqrt((1+4+9+16)/4 + eps).
        let rms = (30.0f32 / 4.0 + eps).sqrt();
        let expected: Vec<f32> = x.iter().map(|v| v / rms).collect();
        let got = rmsnorm(&x, &w, eps);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        // A non-unit gain scales the output linearly.
        let w2 = vec![2.0f32; 4];
        let got2 = rmsnorm(&x, &w2, eps);
        for (g2, g1) in got2.iter().zip(got.iter()) {
            assert!((g2 - 2.0 * g1).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0, 100.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Monotonicity is preserved.
        assert!(v[4] > v[3] && v[3] > v[2]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_dot_matches_scalar() {
        if !has_avx2_fma() {
            return; // non-AVX2 machine: nothing to compare
        }
        // Lengths deliberately not multiples of 16 to exercise the tail loop.
        for n in [1usize, 7, 8, 15, 16, 33, 64, 100, 1024] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
            let scalar = dot_scalar(&a, &b);
            // SAFETY: gated on has_avx2_fma(); slices have equal length n.
            let simd = unsafe { dot_avx2(&a, &b) };
            let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
            assert!(rel < 1e-5, "n={n}: scalar {scalar} vs avx2 {simd}");
        }
    }
}
