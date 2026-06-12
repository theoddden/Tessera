use crate::adapter::weights::AdapterStore;
use crate::api::models::GenerationContext;
use crate::cache::semantic::SemanticCache;
use crate::error::TesseraError;
use crate::generation::client::HypernetworkClient;
use ndarray::{ArrayD, IxDyn};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AtomicSkill {
    pub skill_id: String,
    pub skill_name: String,
    pub domain: String,
    pub embedding: Vec<f32>,
    pub adapter_path: String,
}

#[derive(Debug)]
pub struct CompositionResult {
    pub adapter_id: String,
    pub composed_weights: Vec<u8>,
    pub skill_mix: Vec<(String, f32)>, // (skill_id, weight)
    pub base_model: String,
    pub rank: u32,
}

pub struct SkillMixer {
    adapter_store: Arc<AdapterStore>,
    semantic_cache: Arc<SemanticCache>,
    hypernetwork: Arc<HypernetworkClient>,
    skill_library: Arc<tokio::sync::RwLock<HashMap<String, AtomicSkill>>>,
}

impl SkillMixer {
    pub fn new(
        adapter_store: Arc<AdapterStore>,
        semantic_cache: Arc<SemanticCache>,
        hypernetwork: Arc<HypernetworkClient>,
    ) -> Self {
        SkillMixer {
            adapter_store,
            semantic_cache,
            hypernetwork,
            skill_library: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_skill(
        &self,
        skill_id: String,
        skill_name: String,
        domain: String,
        embedding: Vec<f32>,
        adapter_path: String,
    ) -> Result<(), TesseraError> {
        let mut library = self.skill_library.write().await;
        library.insert(
            skill_id.clone(),
            AtomicSkill {
                skill_id,
                skill_name,
                domain,
                embedding,
                adapter_path,
            },
        );
        Ok(())
    }

    pub async fn compose(
        &self,
        context: &GenerationContext,
        base_model: &str,
        rank: u32,
    ) -> Result<CompositionResult, TesseraError> {
        // 1. Semantic lookup: find relevant atomic skills
        let relevant_skills = self.find_relevant_skills(context).await?;

        if relevant_skills.is_empty() {
            return Err(TesseraError::HypernetworkError(
                "No relevant atomic skills found".to_string(),
            ));
        }

        // 2. Hypernetwork predicts mixing weights
        let mixing_weights = self
            .predict_mixing_weights(context, &relevant_skills)
            .await?;

        // 3. Load atomic skill adapters
        let skill_adapters = self.load_skill_adapters(&relevant_skills).await?;

        // 4. Compose: weighted sum of adapter matrices
        let composed_weights =
            self.compose_adapters(&skill_adapters, &mixing_weights, &relevant_skills)?;

        // 5. Serialize composed adapter
        let adapter_id = uuid::Uuid::new_v4().to_string();
        let composed_bytes = self.serialize_composed(&composed_weights, rank)?;

        // 6. Save composed adapter
        let _ = self
            .adapter_store
            .save(&adapter_id, &composed_bytes)
            .await?;

        let skill_mix: Vec<(String, f32)> = relevant_skills
            .iter()
            .zip(mixing_weights.iter())
            .map(|(skill, weight)| (skill.skill_id.clone(), *weight))
            .collect();

        Ok(CompositionResult {
            adapter_id,
            composed_weights: composed_bytes,
            skill_mix,
            base_model: base_model.to_string(),
            rank,
        })
    }

    async fn find_relevant_skills(
        &self,
        _context: &GenerationContext,
    ) -> Result<Vec<AtomicSkill>, TesseraError> {
        let library = self.skill_library.read().await;

        // For now, return all skills as relevant
        // In production, this would use semantic similarity
        let skills: Vec<AtomicSkill> = library.values().cloned().collect();

        Ok(skills)
    }

    async fn predict_mixing_weights(
        &self,
        _context: &GenerationContext,
        skills: &[AtomicSkill],
    ) -> Result<Vec<f32>, TesseraError> {
        // Simple heuristic: equal weights for now
        // In production, this would call the hypernetwork to predict optimal weights
        let weight = 1.0 / skills.len() as f32;
        Ok(vec![weight; skills.len()])
    }

    async fn load_skill_adapters(
        &self,
        skills: &[AtomicSkill],
    ) -> Result<HashMap<String, Vec<u8>>, TesseraError> {
        let mut adapters = HashMap::new();

        for skill in skills {
            let bytes = self.adapter_store.load(&skill.adapter_path).await?;
            adapters.insert(skill.skill_id.clone(), bytes);
        }

        Ok(adapters)
    }

    fn compose_adapters(
        &self,
        skill_adapters: &HashMap<String, Vec<u8>>,
        mixing_weights: &[f32],
        skills: &[AtomicSkill],
    ) -> Result<HashMap<String, ArrayD<f32>>, TesseraError> {
        let mut composed = HashMap::new();

        // Get tensor names from first adapter
        let first_bytes = skill_adapters.values().next().unwrap();
        let first_tensors = SafeTensors::deserialize(first_bytes)
            .map_err(|e| TesseraError::CorruptAdapter(e.to_string()))?;
        let tensor_names: Vec<String> = first_tensors
            .tensors()
            .into_iter()
            .map(|(name, _)| name.to_string())
            .collect();

        for tensor_name in &tensor_names {
            let mut composed_tensor = None;

            // Iterate over skills and weights together (deterministic order)
            for (skill, weight) in skills.iter().zip(mixing_weights.iter()) {
                if let Some(bytes) = skill_adapters.get(&skill.skill_id) {
                    let tensors = SafeTensors::deserialize(bytes)
                        .map_err(|e| TesseraError::CorruptAdapter(e.to_string()))?;
                    if let Ok(tensor) = tensors.tensor(tensor_name) {
                        use safetensors::tensor::Dtype;
                        if tensor.dtype() != Dtype::F32 {
                            continue;
                        }
                        let data: &[u8] = tensor.data();
                        let shape = tensor.shape();
                        let tensor_array: ArrayD<f32> = ArrayD::from_shape_vec(
                            IxDyn(shape),
                            bytemuck::cast_slice(data).to_vec(),
                        )
                        .map_err(|e| TesseraError::InvalidAdapter(e.to_string()))?;

                        let weighted = &tensor_array * *weight;

                        composed_tensor = Some(match composed_tensor {
                            None => weighted,
                            Some(existing) => existing + weighted,
                        });
                    }
                }
            }

            if let Some(tensor) = composed_tensor {
                composed.insert(tensor_name.clone(), tensor);
            }
        }

        Ok(composed)
    }

    fn serialize_composed(
        &self,
        composed: &HashMap<String, ArrayD<f32>>,
        rank: u32,
    ) -> Result<Vec<u8>, TesseraError> {
        use safetensors::tensor::{Dtype, TensorView};

        // Collect all data vectors first to ensure they live long enough
        let mut data_vecs: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();

        for (name, tensor) in composed {
            let data: Vec<f32> = tensor.iter().cloned().collect();
            let shape: Vec<usize> = tensor.shape().to_vec();
            data_vecs.push((name.clone(), data, shape));
        }

        // Create TensorViews from the collected data
        let mut tensors: Vec<(String, TensorView<'_>)> = Vec::new();
        for (name, data, shape) in &data_vecs {
            let tensor_view =
                TensorView::new(Dtype::F32, shape.clone(), bytemuck::cast_slice(data))
                    .map_err(|e| TesseraError::InvalidAdapter(e.to_string()))?;
            tensors.push((name.clone(), tensor_view));
        }

        safetensors::serialize(tensors.into_iter(), &Default::default()).map_err(|e| {
            TesseraError::SerializationError(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))
        })
    }

    pub async fn list_skills(&self) -> Result<Vec<AtomicSkill>, TesseraError> {
        let library = self.skill_library.read().await;
        Ok(library.values().cloned().collect())
    }
}
