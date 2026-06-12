use crate::api::models::GenerationContext;
use crate::error::TesseraError;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_transformers::models::bert::{BertModel, Config as BertConfig, Config};
use candle_transformers::quantized::gguf_file;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json::Value;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct Encoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Encoder {
    pub async fn new(model_id: &str) -> Result<Self, TesseraError> {
        let api = Api::new()?;
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "refs/pr/15".to_string(),
        );

        let config_path = api
            .model(&repo)
            .get("config.json")
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let tokenizer_path = api
            .model(&repo)
            .get("tokenizer.json")
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let model_path = api
            .model(&repo)
            .get("model.safetensors")
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        // Use CUDA if available, otherwise CPU
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let config: BertConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?,
        )
        .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let vb = candle_nn::VarBuilder::from_safetensors(
            &[model_path],
            candle_core::DType::F32,
            &device,
        )
        .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let model = BertModel::load(vb, &config)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        Ok(Encoder {
            model,
            tokenizer,
            device,
        })
    }

    pub async fn encode(&self, context: &GenerationContext) -> Result<Vec<f32>, TesseraError> {
        let text = self.serialize_context(context);

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let output = self
            .model
            .forward(&input_ids, &attention_mask)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let embedding = self
            .mean_pool(&output, &attention_mask)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        let normalized = self
            .l2_normalize(&embedding)
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?;

        Ok(normalized
            .to_vec1::<f32>()
            .map_err(|e| TesseraError::EmbeddingError(e.to_string()))?)
    }

    fn serialize_context(&self, ctx: &GenerationContext) -> String {
        let mut parts = Vec::new();

        if let Some(desc) = &ctx.description {
            parts.push(desc.clone());
        }
        if let Some(domain) = &ctx.domain {
            parts.push(format!("domain: {}", domain));
        }
        if let Some(meta) = &ctx.metadata {
            if let Some(keys) = meta.as_object() {
                let summary: Vec<String> = keys.keys().take(10).map(|k| k.clone()).collect();
                parts.push(format!("fields: {}", summary.join(", ")));
            }
        }
        if let Some(docs) = &ctx.documents {
            if let Some(first) = docs.first() {
                parts.push(first.chars().take(200).collect::<String>());
            }
        }

        parts.join(" | ")
    }

    fn mean_pool(&self, output: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let attention_mask_expanded = attention_mask
            .unsqueeze(2)
            .expand(output.dims())?;

        let sum = (output * attention_mask_expanded)?.sum(1)?;
        let mask_sum = attention_mask_expanded.sum(1)?;
        let mean = sum.broadcast_div(&mask_sum)?;

        Ok(mean)
    }

    fn l2_normalize(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        let norm = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
        tensor.broadcast_div(&norm)
    }
}
