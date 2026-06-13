use crate::adapter::validate::AdapterValidator;
use crate::adapter::weights::AdapterStore;
use crate::api::models::GenerationContext;
use crate::cache::store::CacheStore;
use crate::error::TesseraError;
use crate::generation::client::HypernetworkClient;
use std::sync::Arc;

#[derive(Debug)]
pub struct GenerationResult {
    pub adapter_id: String,
    pub adapter_path: String,
    pub weights: Vec<u8>,
    pub rank: u32,
    pub archetype_id: String,
    pub archetype_label: String,
    pub source_type: String,
    pub estimated_quality: f32,
    pub recommended_vllm_args: Vec<String>,
}

pub struct GenerationPipeline {
    pub hypernetwork: Arc<HypernetworkClient>,
    adapter_store: Arc<AdapterStore>,
    cache_store: Arc<CacheStore>,
}

impl GenerationPipeline {
    pub fn new(
        hypernetwork: Arc<HypernetworkClient>,
        adapter_store: Arc<AdapterStore>,
        cache_store: Arc<CacheStore>,
    ) -> Self {
        GenerationPipeline {
            hypernetwork,
            adapter_store,
            cache_store,
        }
    }

    pub async fn generate(
        &self,
        context: &GenerationContext,
        base_model: &str,
        rank: u32,
    ) -> Result<GenerationResult, TesseraError> {
        // Call hypernetwork service
        let raw = self
            .hypernetwork
            .generate(context, base_model, rank)
            .await?;

        // Validate adapter shape BEFORE saving
        AdapterValidator::validate_shape(&raw.bytes, base_model, raw.rank)?;

        // Estimate quality
        let quality = AdapterValidator::estimate_quality(&raw.bytes)?;

        // Generate adapter ID
        let adapter_id = uuid::Uuid::new_v4().to_string();

        // Save adapter to store
        let adapter_path = self.adapter_store.save(&adapter_id, &raw.bytes).await?;

        // Record in cache store (cleanup file if this fails)
        let cache_result = self
            .cache_store
            .record_adapter(
                &adapter_id,
                adapter_path.to_str().unwrap(),
                base_model,
                rank,
                &raw.source_type,
            )
            .await;

        if let Err(e) = cache_result {
            // Cleanup: delete the saved file if cache recording failed
            let _ = self.adapter_store.delete(&adapter_id).await;
            return Err(e);
        }

        // Build vLLM args
        let vllm_args = self.build_vllm_args(rank);

        Ok(GenerationResult {
            adapter_id,
            adapter_path: adapter_path.to_str().unwrap().to_string(),
            weights: raw.bytes,
            rank,
            archetype_id: uuid::Uuid::new_v4().to_string(),
            archetype_label: self.auto_label(&raw.source_type),
            source_type: raw.source_type,
            estimated_quality: quality,
            recommended_vllm_args: vllm_args,
        })
    }

    fn build_vllm_args(&self, rank: u32) -> Vec<String> {
        vec![
            "--enable-lora".to_string(),
            format!("--max-lora-rank {}", rank),
            "--lora-modules".to_string(),
        ]
    }

    fn auto_label(&self, source_type: &str) -> String {
        match source_type {
            "doc" => "document_adapter".to_string(),
            "text" => "text_adapter".to_string(),
            "metadata" => "metadata_adapter".to_string(),
            _ => "general_adapter".to_string(),
        }
    }
}
