use crate::api::models::GenerationContext;
use crate::error::TesseraError;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum GenerationMode {
    Document,
    Text,
    Metadata,
    Combined,
}

impl std::fmt::Display for GenerationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationMode::Document => write!(f, "doc"),
            GenerationMode::Text => write!(f, "text"),
            GenerationMode::Metadata => write!(f, "metadata"),
            GenerationMode::Combined => write!(f, "combined"),
        }
    }
}

#[derive(Debug)]
pub struct RawAdapterWeights {
    pub bytes: Vec<u8>,
    pub rank: u32,
    #[allow(dead_code)]
    pub base_model: String,
    pub source_type: String,
}

pub struct HypernetworkClient {
    client: Client,
    base_url: String,
    timeout: Duration,
}

impl HypernetworkClient {
    pub fn new(base_url: &str, timeout_ms: u64) -> Self {
        HypernetworkClient {
            client: Client::new(),
            base_url: base_url.to_string(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    pub async fn generate(
        &self,
        context: &GenerationContext,
        base_model: &str,
        rank: u32,
    ) -> Result<RawAdapterWeights, TesseraError> {
        let mode = self.infer_mode(context);

        let request_body = json!({
            "model": "hypernetwork",
            "messages": [{
                "role": "user",
                "content": self.format_prompt(context, mode.clone())
            }],
            "base_model": base_model,
            "target_rank": rank,
            "response_format": {"type": "lora_weights"},
            "mode": mode.to_string()
        });

        let response = self
            .client
            .post(format!("{}/v1/generate", self.base_url))
            .json(&request_body)
            .timeout(self.timeout)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(TesseraError::HypernetworkError(response.text().await?));
        }

        // Stream adapter weights directly from response body
        let bytes = response.bytes().await?;

        // Parse actual rank from lora_A tensor shape instead of trusting the requested rank
        let actual_rank = safetensors::SafeTensors::deserialize(&bytes)
            .ok()
            .and_then(|tensors| {
                tensors
                    .tensors()
                    .into_iter()
                    .find(|(name, _)| name.contains("lora_A"))
                    .and_then(|(_, t)| t.shape().first().copied())
                    .map(|r| r as u32)
            })
            .unwrap_or(rank);

        Ok(RawAdapterWeights {
            bytes: bytes.to_vec(),
            rank: actual_rank,
            base_model: base_model.to_string(),
            source_type: mode.to_string(),
        })
    }

    pub async fn health_check(&self) -> Result<bool, TesseraError> {
        let response = self
            .client
            .get(format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    fn infer_mode(&self, ctx: &GenerationContext) -> GenerationMode {
        match (&ctx.documents, &ctx.description, &ctx.metadata) {
            (Some(_), None, None) => GenerationMode::Document,
            (None, Some(_), None) => GenerationMode::Text,
            (None, None, Some(_)) => GenerationMode::Metadata,
            _ => GenerationMode::Combined,
        }
    }

    fn format_prompt(&self, ctx: &GenerationContext, mode: GenerationMode) -> String {
        match mode {
            GenerationMode::Document => ctx
                .documents
                .as_ref()
                .map(|docs| docs.join("\n\n"))
                .unwrap_or_default(),
            GenerationMode::Text => ctx.description.clone().unwrap_or_default(),
            GenerationMode::Metadata => serde_json::to_string_pretty(
                ctx.metadata.as_ref().unwrap_or(&serde_json::Value::Null),
            )
            .unwrap_or_default(),
            GenerationMode::Combined => format!(
                "{}\n\nContext: {}",
                ctx.description.clone().unwrap_or_default(),
                serde_json::to_string(ctx.metadata.as_ref().unwrap_or(&serde_json::Value::Null))
                    .unwrap_or_default()
            ),
        }
    }
}
