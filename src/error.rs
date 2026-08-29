//! Error type shared by all minq modules.

/// Unified error type for the engine.
#[derive(Debug, thiserror::Error)]
pub enum MinqError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("model format error: {0}")]
    Format(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("safetensors error: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, MinqError>;
