use crate::api::models::GenerationContext;
use crate::cache::semantic::SemanticCache;
use crate::error::TesseraError;
use crate::generation::pipeline::GenerationPipeline;
use chrono::{DateTime, Duration, Utc};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct SessionEvent {
    #[allow(dead_code)]
    pub archetype_id: String,
    #[allow(dead_code)]
    pub timestamp: DateTime<Utc>,
    pub domain: String,
}

pub struct PredictivePrefetcher {
    history: Arc<RwLock<VecDeque<SessionEvent>>>,
    horizon_minutes: u32,
    top_k: usize,
    pipeline: Option<Arc<GenerationPipeline>>,
}

impl PredictivePrefetcher {
    pub fn new(horizon_minutes: u32, top_k: usize) -> Self {
        PredictivePrefetcher {
            history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            horizon_minutes,
            top_k,
            pipeline: None,
        }
    }

    pub fn set_pipeline(&mut self, pipeline: Arc<GenerationPipeline>) {
        self.pipeline = Some(pipeline);
    }

    pub async fn run_background(
        self: Arc<Self>,
        _cache: Arc<SemanticCache>,
    ) -> Result<(), TesseraError> {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));

        loop {
            interval.tick().await;

            match self.predict_and_prefetch().await {
                Ok(prefetched) => {
                    tracing::info!(
                        "Prefetched {} archetypes for next {} minutes",
                        prefetched.len(),
                        self.horizon_minutes
                    );
                }
                Err(e) => {
                    tracing::warn!("Prefetch cycle failed: {}", e);
                }
            }
        }
    }

    async fn predict_and_prefetch(&self) -> Result<Vec<String>, TesseraError> {
        let history = self.history.read().await;

        if history.is_empty() {
            return Ok(Vec::new());
        }

        let mut domain_counts: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        for event in history.iter() {
            *domain_counts.entry(event.domain.clone()).or_insert(0) += 1;
        }

        let mut sorted: Vec<(String, u64)> = domain_counts.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

        let top_domains: Vec<String> = sorted
            .into_iter()
            .take(self.top_k)
            .map(|(domain, _)| domain)
            .collect();

        if let Some(pipeline) = &self.pipeline {
            for domain in &top_domains {
                let context = GenerationContext {
                    documents: None,
                    description: Some(format!("{} domain specialist", domain)),
                    metadata: None,
                    domain: Some(domain.clone()),
                };
                let base_model = "meta-llama/Llama-3-8B";
                match pipeline.generate(&context, base_model, 16).await {
                    Ok(result) => {
                        tracing::info!(
                            "Prefetched adapter {} for domain {}",
                            result.adapter_id,
                            domain
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to prefetch for domain {}: {}", domain, e);
                    }
                }
            }
        }

        Ok(top_domains)
    }

    #[allow(dead_code)]
    pub async fn record(&self, event: SessionEvent) {
        let mut history = self.history.write().await;
        history.push_back(event);

        // Keep sliding window of 24 hours
        let cutoff = Utc::now() - Duration::hours(24);
        while history
            .front()
            .map(|e| e.timestamp < cutoff)
            .unwrap_or(false)
        {
            history.pop_front();
        }
    }
}
