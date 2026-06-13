use crate::error::TesseraError;
use qdrant_client::qdrant::{Condition, Filter, PointStruct, SearchPoints, Value};
use serde_json::json;
use std::sync::Arc;
use tokio_rusqlite::Connection;

#[derive(Debug, Clone)]
pub struct CacheHit {
    pub adapter_id: String,
    pub adapter_path: String,
    pub archetype_id: String,
    pub label: String,
    pub rank: u32,
    pub similarity: f32,
    pub vllm_args: Vec<String>,
}

pub struct SemanticCache {
    client: QdrantClient,
    collection: String,
    threshold: f32,
    embedding_dim: u64,
    db: Arc<Connection>,
    db_path: String,
}

impl SemanticCache {
    pub async fn new(
        qdrant_url: &str,
        threshold: f32,
        embedding_dim: u64,
        db_path: &str,
    ) -> Result<Self, TesseraError> {
        let client = QdrantClient::from_url(qdrant_url).build()?;

        let db = Arc::new(
            Connection::open(db_path)
                .await
                .map_err(|e| TesseraError::DatabaseError(e))?,
        );

        let cache = SemanticCache {
            client,
            collection: "tessera_adapters".to_string(),
            threshold,
            embedding_dim,
            db,
            db_path: db_path.to_string(),
        };

        cache.init_collection().await?;
        cache.init_db().await?;

        Ok(cache)
    }

    async fn init_collection(&self) -> Result<(), TesseraError> {
        let collections = self.client.list_collections().await?;

        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.collection);

        if !exists {
            self.client
                .create_collection(
                    &qdrant_client::qdrant::CreateCollection {
                        collection_name: self.collection.clone(),
                        vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                            config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                                qdrant_client::qdrant::VectorParams {
                                    size: self.embedding_dim,
                                    distance: qdrant_client::qdrant::Distance::Cosine.into(),
                                    ..Default::default()
                                },
                            )),
                        }),
                        ..Default::default()
                    },
                    None,
                )
                .await
                .map_err(|e| TesseraError::QdrantError(e.to_string()))?;
        }

        Ok(())
    }

    async fn init_db(&self) -> Result<(), TesseraError> {
        self.db
            .call(move |conn| {
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache_stats (
                        id INTEGER PRIMARY KEY,
                        total_requests INTEGER DEFAULT 0,
                        cache_hits INTEGER DEFAULT 0,
                        hit_rate_1h REAL DEFAULT 0.0
                    )",
                    [],
                )?;
                conn.execute(
                    "INSERT OR IGNORE INTO cache_stats (id, total_requests, cache_hits, hit_rate_1h) 
                     VALUES (1, 0, 0, 0.0)",
                    [],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| TesseraError::DatabaseError(e))?;

        Ok(())
    }

    pub async fn lookup(
        &self,
        embedding: &[f32],
        base_model: &str,
    ) -> Result<Option<CacheHit>, TesseraError> {
        let filter = Filter::must([Condition::matches("base_model", base_model.to_string())]);

        let results = self
            .client
            .search_points(&SearchPoints {
                collection_name: self.collection.clone(),
                vector: embedding.to_vec(),
                filter: Some(filter),
                limit: 1,
                with_payload: Some(true.into()),
                score_threshold: Some(self.threshold),
                ..Default::default()
            })
            .await
            .map_err(|e| TesseraError::QdrantError(e.to_string()))?;

        if results.result.is_empty() {
            return Ok(None);
        }

        let hit = &results.result[0];
        let payload = &hit.payload;

        Ok(Some(CacheHit {
            adapter_id: extract_string(payload, "adapter_id"),
            adapter_path: extract_string(payload, "adapter_path"),
            archetype_id: extract_string(payload, "archetype_id"),
            label: extract_string(payload, "label"),
            rank: extract_u32(payload, "rank"),
            similarity: hit.score,
            vllm_args: extract_string_vec(payload, "vllm_args"),
        }))
    }

    pub async fn store(
        &self,
        embedding: &[f32],
        adapter_id: &str,
        adapter_path: &str,
        base_model: &str,
        rank: u32,
        source_type: &str,
        vllm_args: &[String],
    ) -> Result<(), TesseraError> {
        let archetype_id = uuid::Uuid::new_v4().to_string();
        let label = self.auto_label(source_type);

        let payload: Payload = json!({
            "adapter_id": adapter_id,
            "adapter_path": adapter_path,
            "archetype_id": archetype_id,
            "label": label,
            "base_model": base_model,
            "rank": rank,
            "source_type": source_type,
            "vllm_args": vllm_args,
            "hit_count": 0u64,
            "avg_quality": 0.8f32,
            "created_at": chrono::Utc::now().to_rfc3339(),
        })
        .try_into()
        .map_err(|e| TesseraError::SerializationError(e))?;

        self.client
            .upsert_points_simple(
                &self.collection,
                vec![PointStruct::new(
                    uuid::Uuid::new_v4().to_string(),
                    embedding.to_vec(),
                    payload,
                )],
                None,
            )
            .await
            .map_err(|e| TesseraError::QdrantError(e.to_string()))?;

        Ok(())
    }

    pub async fn record_hit(&self, archetype_id: &str) -> Result<(), TesseraError> {
        let collection = self.collection.clone();
        let client = self.client.clone();
        let id = archetype_id.to_string();

        tokio::spawn(async move {
            // Read-modify-write pattern for proper increment
            // First, retrieve current hit_count
            let points_result = client
                .retrieve_points(&collection, None, vec![id.clone().into()], true, None)
                .await;

            if let Ok(points) = points_result {
                if let Some(point) = points.result.first() {
                    let current_count = point
                        .payload
                        .get("hit_count")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0) as i64;

                    let new_count = current_count + 1;

                    // Update with incremented value
                    let _ = client
                        .set_payload(
                            &collection,
                            None,
                            &qdrant_client::qdrant::SetPayload {
                                payload: Some(
                                    json!({"hit_count": new_count})
                                        .try_into()
                                        .unwrap_or_default(),
                                ),
                                points: Some(qdrant_client::qdrant::PointsSelector {
                                    points_selector_one_of: Some(
                                        qdrant_client::qdrant::points_selector::PointsSelectorOneof::Points(
                                            qdrant_client::qdrant::PointIdsList {
                                                ids: vec![id.into()],
                                            },
                                        ),
                                    ),
                                }),
                            },
                            None,
                        )
                        .await;
                }
            }
        });

        Ok(())
    }

    pub async fn evict_low_quality(
        &self,
        _min_hits: u64,
        _min_quality: f32,
    ) -> Result<u64, TesseraError> {
        // TODO: Implement eviction logic
        Ok(0)
    }

    pub async fn mark_prefetch_priority(&self, _archetypes: &[String]) -> Result<(), TesseraError> {
        // TODO: Mark archetypes for prefetch
        Ok(())
    }

    pub async fn is_connected(&self) -> bool {
        self.client.list_collections().await.is_ok()
    }

    fn auto_label(&self, source_type: &str) -> String {
        match source_type {
            "doc" => "document".to_string(),
            "text" => "text".to_string(),
            "metadata" => "metadata".to_string(),
            _ => "general".to_string(),
        }
    }
}

fn extract_string(payload: &Payload, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

fn extract_u32(payload: &Payload, key: &str) -> u32 {
    payload.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as u32
}

fn extract_string_vec(payload: &Payload, key: &str) -> Vec<String> {
    payload
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}
