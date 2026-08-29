//! `.minq` — minq's self-contained (optionally quantized) model format.
//!
//! Binary layout:
//!
//! ```text
//! offset  size      field
//! 0       8         magic "MINQ0001" (legacy "MINFER01" is also accepted)
//! 8       4         u32 LE, length of the config JSON
//! 12      ..        config JSON (serialized `ModelConfig`)
//! ..      ..        tensor records until EOF:
//!                     u32 name_len | name bytes (UTF-8)
//!                     u8  storage tag: 0 = f32, 1 = q8_0, 2 = q4_0
//!                     u8  ndim | ndim x u64 LE shape
//!                     u64 data byte length | data bytes
//! ```
//!
//! f32 tensors are stored as little-endian `f32`; quantized tensors as the
//! raw block bytes defined in [`crate::quantize`].

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::model::ModelConfig;
use crate::quantize::{QuantDtype, QuantizedTensor};
use crate::tensor::Tensor;
use crate::{MinqError, Result};

/// Magic bytes at the start of every `.minq` file.
pub const MAGIC: [u8; 8] = *b"MINQ0001";

/// Magic of the pre-rename `.minfer` format. The binary layout is identical;
/// readers accept it so quantized models exported before the rename keep
/// working. Writers never emit it.
pub const LEGACY_MAGIC: [u8; 8] = *b"MINFER01";

const TAG_F32: u8 = 0;
const TAG_Q8_0: u8 = 1;
const TAG_Q4_0: u8 = 2;

/// A named weight as stored in a `.minq` file: either dense f32 or quantized.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightTensor {
    F32(Tensor),
    Quant(QuantizedTensor),
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Write the file header (magic + config JSON). First half of the streaming
/// export API: callers that quantize tensors one at a time can write the
/// header up front and then append each record with [`write_minq_record`],
/// keeping peak memory at one tensor instead of the whole packed model.
pub fn write_minq_header(w: &mut impl Write, config: &ModelConfig) -> Result<()> {
    w.write_all(&MAGIC)?;
    let cfg = serde_json::to_vec(config)?;
    w.write_all(&(cfg.len() as u32).to_le_bytes())?;
    w.write_all(&cfg)?;
    Ok(())
}

/// Append one tensor record. Byte layout is exactly what [`write_minq`]
/// emits per record, so streaming and batch exports are interchangeable.
pub fn write_minq_record(w: &mut impl Write, name: &str, tensor: &WeightTensor) -> Result<()> {
    w.write_all(&(name.len() as u32).to_le_bytes())?;
    w.write_all(name.as_bytes())?;
    let (tag, shape, data): (u8, Vec<usize>, &[u8]) = match tensor {
        WeightTensor::F32(t) => {
            let bytes: &[u8] = bytemuck_f32(&t.data);
            (TAG_F32, t.shape.clone(), bytes)
        }
        WeightTensor::Quant(q) => (
            match q.dtype {
                QuantDtype::Q8_0 => TAG_Q8_0,
                QuantDtype::Q4_0 => TAG_Q4_0,
            },
            vec![q.rows, q.cols],
            &q.data,
        ),
    };
    w.write_all(&[tag])?;
    w.write_all(&[shape.len() as u8])?;
    for dim in &shape {
        w.write_all(&(*dim as u64).to_le_bytes())?;
    }
    w.write_all(&(data.len() as u64).to_le_bytes())?;
    w.write_all(data)?;
    Ok(())
}

/// Serialize a config and its tensors to a `.minq` file.
pub fn write_minq(
    path: &Path,
    config: &ModelConfig,
    tensors: &[(String, WeightTensor)],
) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_minq_header(&mut w, config)?;
    for (name, tensor) in tensors {
        write_minq_record(&mut w, name, tensor)?;
    }
    w.flush()?;
    Ok(())
}

/// Reinterpret an f32 slice as little-endian bytes without depending on
/// `bytemuck`; the platform is assumed little-endian (x86/ARM), which is
/// checked explicitly to stay honest.
fn bytemuck_f32(data: &[f32]) -> &[u8] {
    assert!(cfg!(target_endian = "little"), ".minq requires little-endian");
    // SAFETY: f32 is plain-old-data with no padding; viewing it as bytes is
    // always valid. Endianness of the file format matches the host (checked).
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    }
}

/// Advance the read cursor, rejecting any length field that points beyond
/// the end of the file. Every variable-length field is funneled through here
/// *before* allocating, so a malformed file can neither trigger huge
/// allocations nor integer-overflow the cursor.
fn take(pos: &mut u64, len: u64, file_len: u64, what: &str, path: &Path) -> Result<()> {
    let end = pos.checked_add(len).ok_or_else(|| {
        MinqError::Format(format!("{}: {what} length overflows", path.display()))
    })?;
    if end > file_len {
        return Err(MinqError::Format(format!(
            "{}: {what} claims {len} bytes at offset {}, file is {file_len} bytes",
            path.display(),
            *pos
        )));
    }
    *pos = end;
    Ok(())
}

/// Parse a `.minq` file back into a config and named tensors. Files with the
/// legacy `.minfer` magic are accepted as well (same layout).
pub fn read_minq(path: &Path) -> Result<(ModelConfig, Vec<(String, WeightTensor)>)> {
    let file_len = std::fs::metadata(path)?.len();
    let mut r = BufReader::new(File::open(path)?);
    let mut pos = 0u64;

    take(&mut pos, 8, file_len, "magic", path)?;
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC && magic != LEGACY_MAGIC {
        return Err(MinqError::Format(format!(
            "{}: not a .minq/.minfer file (bad magic)",
            path.display()
        )));
    }

    take(&mut pos, 4, file_len, "config length", path)?;
    let cfg_len = read_u32(&mut r)? as u64;
    take(&mut pos, cfg_len, file_len, "config json", path)?;
    let mut cfg = vec![0u8; cfg_len as usize];
    r.read_exact(&mut cfg)?;
    let config: ModelConfig = serde_json::from_slice(&cfg)?;

    let mut tensors = Vec::new();
    loop {
        // Peek one byte to distinguish clean EOF from a truncated record.
        let mut first = [0u8; 1];
        if r.read(&mut first)? == 0 {
            break;
        }
        take(&mut pos, 4, file_len, "tensor name length", path)?;
        let mut rest = [0u8; 3];
        r.read_exact(&mut rest)?;
        let name_len = u32::from_le_bytes([first[0], rest[0], rest[1], rest[2]]) as u64;

        take(&mut pos, name_len, file_len, "tensor name", path)?;
        let mut name = vec![0u8; name_len as usize];
        r.read_exact(&mut name)?;
        let name = String::from_utf8(name)
            .map_err(|e| MinqError::Format(format!("invalid tensor name: {e}")))?;

        take(&mut pos, 2, file_len, "tensor tag/ndim", path)?;
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        let mut ndim = [0u8; 1];
        r.read_exact(&mut ndim)?;

        take(&mut pos, 8 * ndim[0] as u64, file_len, "tensor shape", path)?;
        let mut shape: Vec<usize> = Vec::with_capacity(ndim[0] as usize);
        for _ in 0..ndim[0] {
            let dim = read_u64(&mut r)?;
            let dim = usize::try_from(dim).map_err(|_| {
                MinqError::Format(format!("tensor `{name}`: dimension {dim} too large"))
            })?;
            shape.push(dim);
        }
        // Element count with overflow checking.
        let numel: usize = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| {
            MinqError::Format(format!("tensor `{name}`: shape {shape:?} overflows usize"))
        })?;

        take(&mut pos, 8, file_len, "tensor data length", path)?;
        let data_len = read_u64(&mut r)?;

        // The declared data length must match shape x storage exactly; this
        // is checked before any allocation.
        let expected: Option<u64> = match tag[0] {
            TAG_F32 => (numel as u64).checked_mul(4),
            TAG_Q8_0 | TAG_Q4_0 => {
                if shape.len() != 2 {
                    return Err(MinqError::Format(format!(
                        "quantized tensor `{name}` must be 2-D, got {shape:?}"
                    )));
                }
                let dtype = if tag[0] == TAG_Q8_0 {
                    QuantDtype::Q8_0
                } else {
                    QuantDtype::Q4_0
                };
                let (rows, cols) = (shape[0] as u64, shape[1] as u64);
                if cols % crate::quantize::BLOCK_SIZE as u64 != 0 {
                    return Err(MinqError::Format(format!(
                        "quantized tensor `{name}`: cols {cols} not block-aligned"
                    )));
                }
                rows.checked_mul(cols / crate::quantize::BLOCK_SIZE as u64)
                    .and_then(|b| b.checked_mul(dtype.block_bytes() as u64))
            }
            other => {
                return Err(MinqError::Format(format!(
                    "tensor `{name}`: unknown storage tag {other}"
                )))
            }
        };
        let expected = expected.ok_or_else(|| {
            MinqError::Format(format!("tensor `{name}`: byte count overflows"))
        })?;
        if data_len != expected {
            return Err(MinqError::Format(format!(
                "tensor `{name}`: data length {data_len} != shape-derived {expected}"
            )));
        }

        take(&mut pos, data_len, file_len, "tensor data", path)?;
        let mut data = vec![0u8; data_len as usize];
        r.read_exact(&mut data)?;

        let tensor = match tag[0] {
            TAG_F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                WeightTensor::F32(Tensor::new(floats, shape)?)
            }
            _ => WeightTensor::Quant(QuantizedTensor {
                dtype: if tag[0] == TAG_Q8_0 {
                    QuantDtype::Q8_0
                } else {
                    QuantDtype::Q4_0
                },
                rows: shape[0],
                cols: shape[1],
                data,
            }),
        };
        tensors.push((name, tensor));
    }
    Ok((config, tensors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    fn test_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            intermediate_size: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            has_qkv_bias: false,
            dtype: "q8_0".to_string(),
        }
    }

    fn sample_tensors() -> Vec<(String, WeightTensor)> {
        let dense = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let w = Tensor::new((0..4 * 64).map(|i| i as f32 * 0.01).collect(), vec![4, 64]).unwrap();
        let quant = QuantizedTensor::from_tensor(&w, QuantDtype::Q8_0).unwrap();
        vec![
            ("dense.weight".to_string(), WeightTensor::F32(dense)),
            ("quant.weight".to_string(), WeightTensor::Quant(quant)),
        ]
    }

    #[test]
    fn minq_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("minq_format_test_{}.minq", std::process::id()));

        let tensors = sample_tensors();
        let cfg = test_config();

        write_minq(&path, &cfg, &tensors).unwrap();
        // The writer must emit the new magic.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..8], &MAGIC);

        let (cfg2, tensors2) = read_minq(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg, cfg2);
        assert_eq!(tensors, tensors2);
    }

    #[test]
    fn legacy_minfer_magic_is_accepted() {
        // A file written by the pre-rename version (magic "MINFER01") must
        // still load: take a new-format file and patch only the magic bytes.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("minq_legacy_test_{}.minfer", std::process::id()));

        let tensors = sample_tensors();
        let cfg = test_config();
        write_minq(&path, &cfg, &tensors).unwrap();
        let mut raw = std::fs::read(&path).unwrap();
        raw[..8].copy_from_slice(&LEGACY_MAGIC);
        std::fs::write(&path, &raw).unwrap();

        let (cfg2, tensors2) = read_minq(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg, cfg2);
        assert_eq!(tensors, tensors2);
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("minq_bad_magic_{}.bin", std::process::id()));
        std::fs::write(&path, b"NOTMINFER junk").unwrap();
        assert!(read_minq(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// Write a valid header (magic + config) followed by attacker-controlled
    /// record bytes, and return the path.
    fn write_crafted(tag: &str, record: &[u8]) -> std::path::PathBuf {
        let cfg = serde_json::to_vec(&test_config()).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&(cfg.len() as u32).to_le_bytes());
        buf.extend_from_slice(&cfg);
        buf.extend_from_slice(record);
        let path = std::env::temp_dir().join(format!(
            "minq_crafted_{}_{}.minq",
            std::process::id(),
            tag
        ));
        std::fs::write(&path, &buf).unwrap();
        path
    }

    fn expect_format_err(tag: &str, record: &[u8]) {
        let path = write_crafted(tag, record);
        let result = read_minq(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "{tag}: expected Err, got Ok");
    }

    #[test]
    fn rejects_oversized_name_len() {
        // name_len = u32::MAX with nothing behind it.
        let record = u32::MAX.to_le_bytes();
        expect_format_err("huge_name_len", &record);
    }

    #[test]
    fn rejects_oversized_data_len() {
        // Valid f32 record header for shape [1] but a u64::MAX data length.
        let mut record = Vec::new();
        record.extend_from_slice(&1u32.to_le_bytes()); // name_len
        record.extend_from_slice(b"a");
        record.push(0); // tag f32
        record.push(1); // ndim
        record.extend_from_slice(&1u64.to_le_bytes()); // shape [1]
        record.extend_from_slice(&u64::MAX.to_le_bytes()); // data_len
        expect_format_err("huge_data_len", &record);
    }

    #[test]
    fn rejects_shape_data_len_mismatch() {
        // f32 [2, 3] needs 24 data bytes, header declares 8.
        let mut record = Vec::new();
        record.extend_from_slice(&1u32.to_le_bytes());
        record.extend_from_slice(b"w");
        record.push(0); // tag f32
        record.push(2); // ndim
        record.extend_from_slice(&2u64.to_le_bytes());
        record.extend_from_slice(&3u64.to_le_bytes());
        record.extend_from_slice(&8u64.to_le_bytes()); // data_len, should be 24
        record.extend_from_slice(&[0u8; 8]);
        expect_format_err("shape_data_mismatch", &record);
    }

    #[test]
    fn rejects_shape_product_overflow() {
        // [usize::MAX, 2] overflows any element-count multiplication.
        let mut record = Vec::new();
        record.extend_from_slice(&3u32.to_le_bytes());
        record.extend_from_slice(b"big");
        record.push(0); // tag f32
        record.push(2); // ndim
        record.extend_from_slice(&u64::MAX.to_le_bytes());
        record.extend_from_slice(&2u64.to_le_bytes());
        record.extend_from_slice(&16u64.to_le_bytes()); // any data_len
        record.extend_from_slice(&[0u8; 16]);
        expect_format_err("shape_overflow", &record);
    }
}
