use crate::api::models::*;
use crate::cache::semantic::SemanticCache;
use crate::cache::store::CacheStore;
use crate::embedding::encoder::Encoder;
use crate::error::TesseraError;
use crate::generation::pipeline::GenerationPipeline;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub cache: Arc<SemanticCache>,
    pub cache_store: Arc<CacheStore>,
    pub pipeline: Arc<GenerationPipeline>,
    pub encoder: Arc<Encoder>,
    pub metrics_handle: PrometheusHandle,
}

pub async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, TesseraError> {
    let start = Instant::now();

    // Generate embedding for semantic cache lookup
    let embed_start = Instant::now();
    let embedding = state.encoder.encode(&req.context).await?;
    let embed_ms = embed_start.elapsed().as_millis() as u64;

    // Check semantic cache
    let cache_hit = state.cache.lookup(&embedding, &req.base_model).await?;

    if let Some(hit) = cache_hit {
        // Cache hit: return cached adapter
        let total_ms = start.elapsed().as_millis() as u64;
        let adapter_bytes = state.cache_store.get_adapter_path(&hit.adapter_id).await?;
        let adapter_payload = if let Some(path) = adapter_bytes {
            let bytes = tokio::fs::read(&path).await?;
            to_payload(bytes, &req.response_format, &hit.adapter_id)
        } else {
            AdapterPayload::Bytes(vec![])
        };

        // Record hit
        let _ = state.cache.record_hit(&hit.archetype_id).await;
        let _ = state.cache_store.increment_hit_count(&hit.adapter_id).await;

        return Ok(Json(GenerateResponse {
            adapter_id: hit.adapter_id,
            adapter: adapter_payload,
            base_model: req.base_model,
            rank: hit.rank,
            cache_hit: true,
            cache_similarity: Some(hit.similarity),
            generation_latency_ms: 0,
            total_latency_ms: total_ms,
            embedding_latency_ms: embed_ms,
            archetype_id: hit.archetype_id,
            archetype_label: hit.label,
            metadata: AdapterMetadata {
                created_at: chrono::Utc::now().to_rfc3339(),
                expires_at: None,
                source_type: "cached".to_string(),
                estimated_quality: 0.8,
                recommended_vllm_args: hit.vllm_args,
            },
        }));
    }

    // Cache miss: generate new adapter
    let gen_start = Instant::now();
    let result = state
        .pipeline
        .generate(&req.context, &req.base_model, req.target_rank.unwrap_or(16))
        .await?;
    let gen_ms = gen_start.elapsed().as_millis() as u64;

    // Store in semantic cache
    let _ = state
        .cache
        .store(
            &embedding,
            &result.adapter_id,
            &result.adapter_path,
            &req.base_model,
            result.rank,
            &result.source_type,
            &result.recommended_vllm_args,
        )
        .await;

    let total_ms = start.elapsed().as_millis() as u64;

    let adapter_payload = to_payload(result.weights, &req.response_format, &result.adapter_id);

    Ok(Json(GenerateResponse {
        adapter_id: result.adapter_id,
        adapter: adapter_payload,
        base_model: req.base_model,
        rank: result.rank,
        cache_hit: false,
        cache_similarity: None,
        generation_latency_ms: gen_ms,
        total_latency_ms: total_ms,
        embedding_latency_ms: embed_ms,
        archetype_id: result.archetype_id,
        archetype_label: result.archetype_label,
        metadata: AdapterMetadata {
            created_at: chrono::Utc::now().to_rfc3339(),
            expires_at: None,
            source_type: result.source_type,
            estimated_quality: result.estimated_quality,
            recommended_vllm_args: result.recommended_vllm_args,
        },
    }))
}

pub async fn retrieve(
    State(state): State<AppState>,
    Path(adapter_id): Path<String>,
) -> Result<Json<AdapterRetrieveResponse>, TesseraError> {
    let adapter_path = state.cache_store.get_adapter_path(&adapter_id).await?;
    let adapter_payload = if let Some(path) = adapter_path {
        let bytes = tokio::fs::read(&path).await?;
        to_payload(bytes, &None, &adapter_id)
    } else {
        return Err(TesseraError::InvalidAdapter(format!(
            "Adapter {} not found",
            adapter_id
        )));
    };

    let metadata = state.cache_store.get_adapter_metadata(&adapter_id).await?;
    let meta = metadata.ok_or_else(|| {
        TesseraError::InvalidAdapter(format!("Metadata for {} not found", adapter_id))
    })?;

    Ok(Json(AdapterRetrieveResponse {
        adapter_id,
        adapter: adapter_payload,
        base_model: meta.base_model,
        rank: meta.rank,
        archetype_id: adapter_id.clone(),
        hit_count: meta.hit_count,
        created_at: meta.created_at,
    }))
}

pub async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, TesseraError> {
    let embed_start = Instant::now();
    let embedding = state.encoder.encode(&req.context).await?;
    let embed_ms = embed_start.elapsed().as_millis() as u64;

    let cache_hit = state.cache.lookup(&embedding, &req.base_model).await?;

    if let Some(hit) = cache_hit {
        Ok(Json(EmbedResponse {
            cache_hit: true,
            similarity: Some(hit.similarity),
            adapter_id: Some(hit.adapter_id),
            archetype_label: Some(hit.label),
            embedding_latency_ms: embed_ms,
        }))
    } else {
        Ok(Json(EmbedResponse {
            cache_hit: false,
            similarity: None,
            adapter_id: None,
            archetype_label: None,
            embedding_latency_ms: embed_ms,
        }))
    }
}

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let qdrant_connected = state.cache.client.list_collections().await.is_ok();
    let hypernetwork_connected = state
        .pipeline
        .hypernetwork
        .health_check()
        .await
        .unwrap_or(false);
    let cache_stats = state.cache_store.get_stats().await.unwrap_or_default();

    let status = if qdrant_connected && hypernetwork_connected {
        "healthy".to_string()
    } else {
        "degraded".to_string()
    };

    Json(HealthResponse {
        status,
        cache_size: cache_stats.total_adapters,
        hit_rate_1h: 0.0,
        avg_generation_latency_ms: 0.0,
        avg_cache_hit_latency_ms: 0.0,
        qdrant_connected,
        hypernetwork_connected,
    })
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics_handle.render();
    (StatusCode::OK, metrics)
}

fn to_payload(bytes: Vec<u8>, format: &Option<ResponseFormat>, adapter_id: &str) -> AdapterPayload {
    match format {
        Some(ResponseFormat::Base64) => AdapterPayload::Base64(STANDARD.encode(&bytes)),
        Some(ResponseFormat::Url) => {
            AdapterPayload::Url(format!("https://tessera.local/adapters/{}", adapter_id))
        }
        _ => AdapterPayload::Bytes(bytes),
    }
}
