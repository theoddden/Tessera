use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use thiserror::Error;

impl From<hf_hub::api::sync::ApiError> for TesseraError {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        TesseraError::HfHubError(err.to_string())
    }
}

impl IntoResponse for TesseraError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            TesseraError::HttpError(e) => (StatusCode::BAD_GATEWAY, e.to_string()),
            TesseraError::SerializationError(e) => {
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            }
            TesseraError::QdrantError(e) => (StatusCode::SERVICE_UNAVAILABLE, e),
            TesseraError::EmbeddingError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e),
            TesseraError::HypernetworkError(e) => (StatusCode::BAD_GATEWAY, e),
            TesseraError::InvalidAdapter(e) => (StatusCode::BAD_REQUEST, e),
            TesseraError::CorruptAdapter(e) => (StatusCode::BAD_REQUEST, e),
            TesseraError::RankMismatch { expected, found } => (
                StatusCode::BAD_REQUEST,
                format!("Rank mismatch: expected {}, found {:?}", expected, found),
            ),
            TesseraError::DatabaseError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            TesseraError::IoError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            TesseraError::ConfigError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e),
            TesseraError::HfHubError(e) => (StatusCode::BAD_GATEWAY, e),
        };

        let body = Json(serde_json::json!({
            "error": message,
        }));

        (status, body).into_response()
    }
}

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

    #[error("HuggingFace API error: {0}")]
    HfHubError(String),
}
