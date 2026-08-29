//! minq — a from-scratch CPU inference engine for LLaMA-family transformers.
//!
//! Written without any existing deep-learning framework: tensors, quantized
//! kernels, the transformer forward pass, sampling and the generation loop
//! are all implemented by hand on top of `rayon` for thread-level parallelism.

pub mod engine;
pub mod format;
pub mod model;
pub mod ppl;
pub mod quantize;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

mod error;
pub use error::{MinqError, Result};
