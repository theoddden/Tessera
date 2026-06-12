use thiserror::Error;

#[derive(Error, Debug)]
pub enum TesseraError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Qdrant error: {0}")]
    QdrantError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Hypernetwork error: {0}")]
    HypernetworkError(String),

    #[error("Invalid adapter: {0}")]
    InvalidAdapter(String),

    #[error("Corrupt adapter: {0}")]
    CorruptAdapter(String),

    #[error("Rank mismatch: expected {expected}, found {found:?}")]
    RankMismatch { expected: u32, found: Vec<usize> },

    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Config error: {0}")]
    ConfigError(String),
}
