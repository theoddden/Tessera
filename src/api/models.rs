use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    #[allow(dead_code)]
    pub user_id: String,
    pub context: GenerationContext,
    pub base_model: String,
    pub target_rank: Option<u32>,
    #[allow(dead_code)]
    pub hypernetwork_url: Option<String>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationContext {
    pub documents: Option<Vec<String>>,
    pub description: Option<String>,
    pub metadata: Option<Value>,
    pub domain: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormat {
    File,
    Base64,
    Url,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub adapter_id: String,
    pub adapter: AdapterPayload,
    pub base_model: String,
    pub rank: u32,
    pub cache_hit: bool,
    pub cache_similarity: Option<f32>,
    pub generation_latency_ms: u64,
    pub total_latency_ms: u64,
    pub embedding_latency_ms: u64,
    pub archetype_id: String,
    pub archetype_label: String,
    pub metadata: AdapterMetadata,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum AdapterPayload {
    Bytes(Vec<u8>),
    Base64(String),
    Url(String),
}

#[derive(Debug, Serialize)]
pub struct AdapterMetadata {
    pub created_at: String,
    pub expires_at: Option<String>,
    pub source_type: String,
    pub estimated_quality: f32,
    pub recommended_vllm_args: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AdapterRetrieveResponse {
    pub adapter_id: String,
    pub adapter: AdapterPayload,
    pub base_model: String,
    pub rank: u32,
    pub archetype_id: String,
    pub hit_count: u64,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub context: GenerationContext,
    pub base_model: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub cache_hit: bool,
    pub similarity: Option<f32>,
    pub adapter_id: Option<String>,
    pub archetype_label: Option<String>,
    pub embedding_latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub cache_size: u64,
    pub hit_rate_1h: f32,
    pub avg_generation_latency_ms: f64,
    pub avg_cache_hit_latency_ms: f64,
    pub qdrant_connected: bool,
    pub hypernetwork_connected: bool,
}
