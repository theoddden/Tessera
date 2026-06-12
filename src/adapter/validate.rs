use crate::error::TesseraError;
use safetensors::SafeTensors;

pub struct AdapterValidator;

impl AdapterValidator {
    pub fn validate_shape(
        bytes: &[u8],
        base_model: &str,
        expected_rank: u32,
    ) -> Result<(), TesseraError> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| TesseraError::InvalidAdapter(e.to_string()))?;

        // Check that all lora_A and lora_B tensors have correct rank dim
        for (name, tensor) in tensors.tensors() {
            if name.contains("lora_A") || name.contains("lora_B") {
                let shape = tensor.shape();
                if !shape.contains(&(expected_rank as usize)) {
                    return Err(TesseraError::RankMismatch {
                        expected: expected_rank,
                        found: shape.to_vec(),
                    });
                }
            }
        }

        // Validate base model dimensions
        Self::validate_model_dimensions(tensors, base_model)?;

        Ok(())
    }

    fn validate_model_dimensions(
        tensors: SafeTensors,
        base_model: &str,
    ) -> Result<(), TesseraError> {
        // Known model dimensions
        let model_dims = match base_model {
            "meta-llama/Llama-3-8B" => (4096, 4096),
            "meta-llama/Llama-3-70B" => (8192, 8192),
            "Qwen/Qwen2-7B" => (3584, 3584),
            "deepseek-ai/DeepSeek-V3" => (7168, 7168),
            _ => return Ok(()), // Skip validation for unknown models
        };

        let (d_in, d_out) = model_dims;

        for (name, tensor) in tensors.tensors() {
            if name.contains("lora_A") {
                let shape = tensor.shape();
                if shape.len() >= 2 && shape[1] != d_in {
                    return Err(TesseraError::InvalidAdapter(format!(
                        "Tensor {} has incompatible input dimension: expected {}, found {:?}",
                        name, d_in, shape
                    )));
                }
            }
            if name.contains("lora_B") {
                let shape = tensor.shape();
                if shape.len() >= 2 && shape[0] != d_out {
                    return Err(TesseraError::InvalidAdapter(format!(
                        "Tensor {} has incompatible output dimension: expected {}, found {:?}",
                        name, d_out, shape
                    )));
                }
            }
        }

        Ok(())
    }

    pub fn estimate_quality(bytes: &[u8]) -> Result<f32, TesseraError> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| TesseraError::InvalidAdapter(e.to_string()))?;

        // Simple heuristic: check weight distribution
        let mut total_norm = 0.0f32;
        let mut count = 0;

        for (_name, tensor) in tensors.tensors() {
            // Check dtype before casting
            use safetensors::tensor::Dtype;
            match tensor.dtype() {
                Dtype::F32 => {
                    if let Ok(data) = tensor.data() {
                        let data: &[f32] = bytemuck::cast_slice(data);
                        let norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                        total_norm += norm;
                        count += 1;
                    }
                }
                Dtype::F16 | Dtype::BF16 => {
                    // Skip half-precision tensors for quality estimation
                    // or convert them properly if needed
                    continue;
                }
                _ => {
                    // Skip other dtypes
                    continue;
                }
            }
        }

        if count == 0 {
            return Ok(0.5); // Default quality
        }

        let avg_norm = total_norm / count as f32;
        
        // Normalize to 0-1 range (heuristic)
        let quality = (avg_norm / 100.0).min(1.0).max(0.0);
        
        Ok(quality)
    }
}
