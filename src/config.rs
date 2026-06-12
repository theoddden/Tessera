use anyhow::Result;
use serde::Deserialize;
use std::env;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub port: u16,
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub similarity_threshold: f32,
    pub embedding_model: String,
    pub embedding_dim: u64,
    pub hypernetwork_url: String,
    pub generation_timeout_ms: u64,
    pub adapter_store_path: String,
    pub prefetch_horizon_minutes: u32,
    pub prefetch_top_k: usize,
    pub log_level: String,
    pub cache_db_path: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Config {
            port: env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()?,
            qdrant_url: env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_collection: env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "tessera_adapters".to_string()),
            similarity_threshold: env::var("SIMILARITY_THRESHOLD")
                .unwrap_or_else(|_| "0.92".to_string())
                .parse()?,
            embedding_model: env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string()),
            embedding_dim: env::var("EMBEDDING_DIM")
                .unwrap_or_else(|_| "384".to_string())
                .parse()?,
            hypernetwork_url: env::var("HYPERNETWORK_URL")
                .unwrap_or_else(|_| "http://localhost:8000".to_string()),
            generation_timeout_ms: env::var("GENERATION_TIMEOUT_MS")
                .unwrap_or_else(|_| "30000".to_string())
                .parse()?,
            adapter_store_path: env::var("ADAPTER_STORE_PATH")
                .unwrap_or_else(|_| "./adapters".to_string()),
            prefetch_horizon_minutes: env::var("PREFETCH_HORIZON_MINUTES")
                .unwrap_or_else(|_| "60".to_string())
                .parse()?,
            prefetch_top_k: env::var("PREFETCH_TOP_K")
                .unwrap_or_else(|_| "10".to_string())
                .parse()?,
            log_level: env::var("LOG_LEVEL")
                .unwrap_or_else(|_| "info".to_string()),
            cache_db_path: env::var("CACHE_DB_PATH")
                .unwrap_or_else(|_| "./cache.db".to_string()),
        })
    }
}
