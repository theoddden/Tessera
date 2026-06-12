mod adapter;
mod api;
mod cache;
mod composition;
mod config;
mod embedding;
mod error;
mod generation;

use adapter::weights::AdapterStore;
use api::routes::{embed, generate, health, metrics, retrieve, AppState};
use cache::prefetch::PredictivePrefetcher;
use cache::semantic::SemanticCache;
use cache::store::CacheStore;
use composition::mixer::SkillMixer;
use config::Config;
use embedding::encoder::Encoder;
use generation::client::HypernetworkClient;
use generation::pipeline::GenerationPipeline;
use metrics::{counter, gauge};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config
    let config = Config::from_env()?;

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(&config.log_level)
        .init();

    tracing::info!("Starting Tessera v0.1.0");

    // Initialize metrics exporter
    let recorder = PrometheusBuilder::new().build();
    let metrics_handle = recorder.handle();
    metrics::set_global_recorder(recorder).expect("Failed to set metrics recorder");

    // Initialize components
    let cache = Arc::new(
        SemanticCache::new(
            &config.qdrant_url,
            config.similarity_threshold,
            config.embedding_dim,
            &config.cache_db_path,
        )
        .await?,
    );

    let cache_store = Arc::new(CacheStore::new(&config.cache_db_path).await?);

    let adapter_store = Arc::new(AdapterStore::new(&config.adapter_store_path)?);

    let hypernetwork = Arc::new(HypernetworkClient::new(
        &config.hypernetwork_url,
        config.generation_timeout_ms,
    ));

    let pipeline = Arc::new(GenerationPipeline::new(
        hypernetwork.clone(),
        adapter_store.clone(),
        cache_store.clone(),
    ));

    let mut prefetcher =
        PredictivePrefetcher::new(config.prefetch_horizon_minutes, config.prefetch_top_k);
    prefetcher.set_pipeline(pipeline.clone());
    let prefetcher = Arc::new(prefetcher);

    let skill_mixer = Arc::new(SkillMixer::new(
        adapter_store.clone(),
        cache.clone(),
        hypernetwork.clone(),
    ));

    // Start background prefetch loop
    let prefetcher_bg = prefetcher.clone();
    let cache_bg = cache.clone();
    tokio::spawn(async move {
        let _ = prefetcher_bg.run_background(cache_bg).await;
    });

    // Start cache eviction loop
    let cache_evict = cache.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
        loop {
            interval.tick().await;
            let _ = cache_evict.evict_low_quality(5, 0.70).await;
        }
    });

    let state = AppState {
        cache,
        cache_store,
        pipeline,
        encoder: Arc::new(Encoder::new(&config.embedding_model).await?),
        metrics_handle,
    };

    // Build router
    let app = axum::Router::new()
        .route("/generate", axum::routing::post(generate))
        .route("/adapter/:adapter_id", axum::routing::get(retrieve))
        .route("/embed", axum::routing::post(embed))
        .route("/health", axum::routing::get(health))
        .route("/metrics", axum::routing::get(metrics))
        .with_state(state)
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // Start server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", config.port)).await?;

    tracing::info!("Tessera listening on port {}", config.port);

    axum::serve(listener, app).await?;

    Ok(())
}
