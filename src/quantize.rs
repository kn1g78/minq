//! Block-quantized weight formats and their fused dequantize-multiply kernels.
//!
//! The layouts follow the GGML family: weights are split into blocks of 32
//! values, and each block stores one `f32` scale plus the quantized codes.
//!
//! - `Q8_0`: 36 bytes/block = 1 x f32 scale + 32 x i8 codes, `x ~ q * d`
//!   with `d = max|x| / 127`.
//! - `Q4_0`: 20 bytes/block = 1 x f32 scale + 32 x 4-bit codes packed into
//!   16 bytes (low nibble = element 2i, high nibble = element 2i+1),
//!   `x ~ (code - 8) * d` with `d = max|x| / 8`.
//!
//! The inference hot path is [`QuantizedTensor::matvec`], which fuses
//! dequantization into the multiply-accumulate loop so weights are never
//! materialized in f32.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;
use crate::{MinqError, Result};

/// Number of weights per quantization block.
pub const BLOCK_SIZE: usize = 32;
/// Serialized size of one Q8_0 block (4-byte scale + 32 i8 codes).
pub const Q8_0_BLOCK_BYTES: usize = 4 + 32;
/// Serialized size of one Q4_0 block (4-byte scale + 16 packed bytes).
pub const Q4_0_BLOCK_BYTES: usize = 4 + 16;

/// Supported quantization dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantDtype {
    Q8_0,
    Q4_0,
}

impl QuantDtype {
    pub fn block_bytes(self) -> usize {
        match self {
            QuantDtype::Q8_0 => Q8_0_BLOCK_BYTES,
            QuantDtype::Q4_0 => Q4_0_BLOCK_BYTES,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            QuantDtype::Q8_0 => "q8_0",
            QuantDtype::Q4_0 => "q4_0",
        }
    }

    /// Parse a dtype name (`q8_0` / `q4_0`), e.g. from the CLI.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "q8_0" => Ok(QuantDtype::Q8_0),
            "q4_0" => Ok(QuantDtype::Q4_0),
            other => Err(MinqError::Format(format!(
                "unknown dtype `{other}` (expected q8_0 or q4_0)"
            ))),
        }
    }
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn check_len(x: &[f32]) -> Result<()> {
    if x.len() % BLOCK_SIZE != 0 {
        return Err(MinqError::Shape(format!(
            "quantization requires a multiple of {BLOCK_SIZE} elements, got {}",
            x.len()
        )));
    }
    Ok(())
}

/// Quantize `x` into the Q8_0 byte layout.
pub fn quantize_q8_0(x: &[f32]) -> Result<Vec<u8>> {
    check_len(x)?;
    let mut out = Vec::with_capacity(x.len() / BLOCK_SIZE * Q8_0_BLOCK_BYTES);
    for block in x.chunks_exact(BLOCK_SIZE) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&d.to_le_bytes());
        if d == 0.0 {
            out.extend_from_slice(&[0u8; BLOCK_SIZE]);
        } else {
            let id = 1.0 / d;
            for v in block {
                let q = (v * id).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
    }
    Ok(out)
}

/// Inverse of [`quantize_q8_0`]; `out.len()` must equal `bytes.len() / 36 * 32`.
pub fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if bytes.len() % Q8_0_BLOCK_BYTES != 0
        || out.len() != bytes.len() / Q8_0_BLOCK_BYTES * BLOCK_SIZE
    {
        return Err(MinqError::Shape(format!(
            "dequantize_q8_0: {} bytes vs {} outputs",
            bytes.len(),
            out.len()
        )));
    }
    for (bi, block) in bytes.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = read_f32_le(block);
        let base = bi * BLOCK_SIZE;
        for (i, &q) in block[4..].iter().enumerate() {
            out[base + i] = (q as i8) as f32 * d;
        }
    }
    Ok(())
}

/// Quantize `x` into the Q4_0 byte layout.
pub fn quantize_q4_0(x: &[f32]) -> Result<Vec<u8>> {
    check_len(x)?;
    let mut out = Vec::with_capacity(x.len() / BLOCK_SIZE * Q4_0_BLOCK_BYTES);
    for block in x.chunks_exact(BLOCK_SIZE) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 8.0;
        out.extend_from_slice(&d.to_le_bytes());
        let mut packed = [0u8; BLOCK_SIZE / 2];
        if d != 0.0 {
            let id = 1.0 / d;
            for (i, v) in block.iter().enumerate() {
                let q = (v * id).round().clamp(-8.0, 7.0) as i32;
                let code = (q + 8) as u8;
                if i % 2 == 0 {
                    packed[i / 2] |= code;
                } else {
                    packed[i / 2] |= code << 4;
                }
            }
        }
        out.extend_from_slice(&packed);
    }
    Ok(out)
}

/// Inverse of [`quantize_q4_0`]; `out.len()` must equal `bytes.len() / 20 * 32`.
pub fn dequantize_q4_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if bytes.len() % Q4_0_BLOCK_BYTES != 0
        || out.len() != bytes.len() / Q4_0_BLOCK_BYTES * BLOCK_SIZE
    {
        return Err(MinqError::Shape(format!(
            "dequantize_q4_0: {} bytes vs {} outputs",
            bytes.len(),
            out.len()
        )));
    }
    for (bi, block) in bytes.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let d = read_f32_le(block);
        let base = bi * BLOCK_SIZE;
        for i in 0..BLOCK_SIZE / 2 {
            let byte = block[4 + i];
            out[base + 2 * i] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            out[base + 2 * i + 1] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
    }
    Ok(())
}

/// A 2-D weight matrix held in block-quantized form, row-major by block.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedTensor {
    pub dtype: QuantDtype,
    pub rows: usize,
    pub cols: usize,
    /// Raw block bytes; row `r` occupies `cols / 32 * block_bytes` bytes.
    pub data: Vec<u8>,
}

impl QuantizedTensor {
    /// Quantize a 2-D f32 weight matrix `[rows, cols]`; `cols` must be a
    /// multiple of the block size.
    pub fn from_tensor(w: &Tensor, dtype: QuantDtype) -> Result<Self> {
        if w.ndim() != 2 || w.shape[1] % BLOCK_SIZE != 0 {
            return Err(MinqError::Shape(format!(
                "cannot quantize shape {:?}: need 2-D with cols % {BLOCK_SIZE} == 0",
                w.shape
            )));
        }
        let data = match dtype {
            QuantDtype::Q8_0 => quantize_q8_0(&w.data)?,
            QuantDtype::Q4_0 => quantize_q4_0(&w.data)?,
        };
        Ok(Self {
            dtype,
            rows: w.shape[0],
            cols: w.shape[1],
            data,
        })
    }

    /// Materialize the weights back to f32 (used for verification, not inference).
    pub fn dequantize(&self) -> Result<Tensor> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        match self.dtype {
            QuantDtype::Q8_0 => dequantize_q8_0(&self.data, &mut out)?,
            QuantDtype::Q4_0 => dequantize_q4_0(&self.data, &mut out)?,
        }
        Tensor::new(out, vec![self.rows, self.cols])
    }

    /// Fused dequantize-multiply: `y = W @ x` without materializing W.
    ///
    /// For each block we accumulate `sum(q_i * x_i)` in integers-cast-to-float
    /// and apply the block scale once, which keeps both the dequantization
    /// work and the memory traffic minimal.
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.cols {
            return Err(MinqError::Shape(format!(
                "quantized matvec: {} cols vs input len {}",
                self.cols,
                x.len()
            )));
        }
        let row_bytes = self.cols / BLOCK_SIZE * self.dtype.block_bytes();
        let dtype = self.dtype;
        let simd = crate::tensor::has_avx2_fma();
        let out: Vec<f32> = self
            .data
            .par_chunks_exact(row_bytes)
            .map(|row| match (dtype, simd) {
                (QuantDtype::Q8_0, false) => matvec_row_q8_0(row, x),
                (QuantDtype::Q4_0, false) => matvec_row_q4_0(row, x),
                #[cfg(target_arch = "x86_64")]
                (QuantDtype::Q8_0, true) => {
                    // SAFETY: AVX2+FMA availability was checked above; the
                    // kernels only read within `row` (chunks_exact guarantees
                    // whole blocks) and `x` (length == cols, checked above).
                    unsafe { avx2::matvec_row_q8_0(row, x) }
                }
                #[cfg(target_arch = "x86_64")]
                (QuantDtype::Q4_0, true) => {
                    // SAFETY: see Q8_0 arm.
                    unsafe { avx2::matvec_row_q4_0(row, x) }
                }
                #[cfg(not(target_arch = "x86_64"))]
                (_, true) => unreachable!("has_avx2_fma() is false off x86_64"),
            })
            .collect();
        Ok(out)
    }
}

fn matvec_row_q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (bi, block) in row.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = read_f32_le(block);
        let base = bi * BLOCK_SIZE;
        let mut sum = 0.0f32;
        for (i, &q) in block[4..].iter().enumerate() {
            sum += (q as i8) as f32 * x[base + i];
        }
        acc += d * sum;
    }
    acc
}

fn matvec_row_q4_0(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (bi, block) in row.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let d = read_f32_le(block);
        let base = bi * BLOCK_SIZE;
        let mut sum = 0.0f32;
        for i in 0..BLOCK_SIZE / 2 {
            let byte = block[4 + i];
            sum += ((byte & 0x0F) as i32 - 8) as f32 * x[base + 2 * i]
                + ((byte >> 4) as i32 - 8) as f32 * x[base + 2 * i + 1];
        }
        acc += d * sum;
    }
    acc
}

/// AVX2+FMA row kernels. Selected at runtime by
/// [`crate::tensor::has_avx2_fma`]; the scalar functions above remain the
/// portable fallback.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::{Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES};
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 packed f32 lanes.
    ///
    /// # Safety
    ///
    /// Requires AVX (implied by the AVX2 callers).
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }

    /// Sign-extend 32 packed i8 to four groups of 8 f32 and accumulate
    /// `sum += q[i] * x[i]` block-wise. Shared tail of both kernels.
    ///
    /// # Safety
    ///
    /// Requires AVX2+FMA; `xp` must be readable for 32 floats.
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn block_dot(q: __m256i, xp: *const f32) -> f32 {
        let lo = _mm256_castsi256_si128(q);
        let hi = _mm256_extracti128_si256(q, 1);
        // _mm256_cvtepi8_epi32 sign-extends the low 8 bytes of its argument.
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        let mut s = _mm256_mul_ps(q0, _mm256_loadu_ps(xp));
        s = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xp.add(8)), s);
        s = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xp.add(16)), s);
        s = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xp.add(24)), s);
        hsum_ps(s)
    }

    /// AVX2+FMA fused dequant-dot of one Q8_0 row with `x`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee AVX2 and FMA are available (checked via
    /// `has_avx2_fma`). `row.len()` must be a multiple of 36 and `x.len()`
    /// must equal `row.len() / 36 * 32`; all loads stay inside the slices
    /// under this contract.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn matvec_row_q8_0(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut rp = row.as_ptr();
        let mut xp = x.as_ptr();
        for _ in 0..row.len() / Q8_0_BLOCK_BYTES {
            let d = f32::from_le_bytes([*rp, *rp.add(1), *rp.add(2), *rp.add(3)]);
            let q = _mm256_loadu_si256(rp.add(4) as *const __m256i);
            acc += d * block_dot(q, xp);
            rp = rp.add(Q8_0_BLOCK_BYTES);
            xp = xp.add(32);
        }
        acc
    }

    /// AVX2+FMA fused dequant-dot of one Q4_0 row with `x`.
    ///
    /// Nibble unpacking: byte i holds element 2i in its low nibble and 2i+1
    /// in its high nibble; `_mm_unpacklo/hi_epi8` interleaves the two nibble
    /// vectors back into sequential order, then 8 is subtracted to center
    /// the codes.
    ///
    /// # Safety
    ///
    /// The caller must guarantee AVX2 and FMA are available. `row.len()`
    /// must be a multiple of 20 and `x.len()` must equal
    /// `row.len() / 20 * 32`; all loads stay inside the slices under this
    /// contract.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn matvec_row_q4_0(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut rp = row.as_ptr();
        let mut xp = x.as_ptr();
        let nibble_mask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        for _ in 0..row.len() / Q4_0_BLOCK_BYTES {
            let d = f32::from_le_bytes([*rp, *rp.add(1), *rp.add(2), *rp.add(3)]);
            let packed = _mm_loadu_si128(rp.add(4) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask); // elements 0,2,4,...
            // No epi8 shift exists; a 16-bit shift + mask extracts high nibbles.
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let e0 = _mm_unpacklo_epi8(lo, hi); // elements 0..15
            let e1 = _mm_unpackhi_epi8(lo, hi); // elements 16..31
            let q = _mm256_sub_epi8(_mm256_set_m128i(e1, e0), eight);
            acc += d * block_dot(q, xp);
            rp = rp.add(Q4_0_BLOCK_BYTES);
            xp = xp.add(32);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (rng.gen::<f32>() - 0.5) * 2.0).collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let err: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let mag: f32 = a.iter().map(|x| x * x).sum();
        (err / mag).sqrt()
    }

    fn roundtrip_q8(x: &[f32]) -> Vec<f32> {
        let bytes = quantize_q8_0(x).unwrap();
        let mut out = vec![0.0; x.len()];
        dequantize_q8_0(&bytes, &mut out).unwrap();
        out
    }

    fn roundtrip_q4(x: &[f32]) -> Vec<f32> {
        let bytes = quantize_q4_0(x).unwrap();
        let mut out = vec![0.0; x.len()];
        dequantize_q4_0(&bytes, &mut out).unwrap();
        out
    }

    #[test]
    fn q8_0_roundtrip_error_is_bounded() {
        // Per-element error is at most d/2 = max|x|/254, so the relative
        // Frobenius error of ~uniform data stays well under 1%.
        let x = rand_vec(32 * 64, 7);
        let y = roundtrip_q8(&x);
        let rel = rel_err(&x, &y);
        assert!(rel < 0.01, "q8_0 relative error {rel} too large");
    }

    #[test]
    fn q4_0_is_less_accurate_than_q8_0() {
        // Sanity monotonicity check: coarser quantization must lose accuracy.
        let x = rand_vec(32 * 64, 11);
        let rel8 = rel_err(&x, &roundtrip_q8(&x));
        let rel4 = rel_err(&x, &roundtrip_q4(&x));
        assert!(
            rel4 > rel8,
            "expected q4_0 error ({rel4}) > q8_0 error ({rel8})"
        );
        // d = max|x|/8 bounds the q4_0 per-element error by max|x|/16.
        assert!(rel4 < 0.25, "q4_0 relative error {rel4} too large");
    }

    #[test]
    fn q8_0_block_uses_full_scale() {
        // A block spanning [-16, 15] must use d = 16/127 exactly.
        let x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let bytes = quantize_q8_0(&x).unwrap();
        let d = read_f32_le(&bytes);
        assert!((d - 16.0 / 127.0).abs() < 1e-9);
        let y = roundtrip_q8(&x);
        let max_err: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err <= d / 2.0 + 1e-6);
    }

    #[test]
    fn quantized_matvec_matches_dequantized_matvec() {
        let w = Tensor::new(rand_vec(4 * 64, 3), vec![4, 64]).unwrap();
        let x = rand_vec(64, 5);
        for dtype in [QuantDtype::Q8_0, QuantDtype::Q4_0] {
            let qt = QuantizedTensor::from_tensor(&w, dtype).unwrap();
            let fast = qt.matvec(&x).unwrap();
            let reference = crate::tensor::matvec(&qt.dequantize().unwrap(), &x).unwrap();
            for (f, r) in fast.iter().zip(reference.iter()) {
                assert!(
                    (f - r).abs() < 1e-3,
                    "{:?}: fused {f} vs dequantized {r}",
                    dtype
                );
            }
        }
    }

    #[test]
    fn rejects_misaligned_input() {
        assert!(quantize_q8_0(&[1.0; 17]).is_err());
        assert!(quantize_q4_0(&[1.0; 33]).is_err());
    }

    /// On this AVX2 machine these tests really exercise the SIMD branch;
    /// elsewhere they no-op (the scalar fallback is already covered above).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q8_0_row_matches_scalar() {
        if !crate::tensor::has_avx2_fma() {
            return;
        }
        let x = rand_vec(32 * 8, 21);
        let row = quantize_q8_0(&rand_vec(32 * 8, 22)).unwrap();
        let scalar = matvec_row_q8_0(&row, &x);
        // SAFETY: gated on has_avx2_fma(); row/x satisfy the length contract.
        let simd = unsafe { avx2::matvec_row_q8_0(&row, &x) };
        let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-5, "scalar {scalar} vs avx2 {simd} (rel {rel})");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q4_0_row_matches_scalar() {
        if !crate::tensor::has_avx2_fma() {
            return;
        }
        let x = rand_vec(32 * 8, 23);
        let row = quantize_q4_0(&rand_vec(32 * 8, 24)).unwrap();
        let scalar = matvec_row_q4_0(&row, &x);
        // SAFETY: gated on has_avx2_fma(); row/x satisfy the length contract.
        let simd = unsafe { avx2::matvec_row_q4_0(&row, &x) };
        let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-5, "scalar {scalar} vs avx2 {simd} (rel {rel})");
    }
}
