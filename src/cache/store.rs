use crate::error::TesseraError;
use rusqlite::{params, Connection};
use std::sync::Arc;
use tokio_rusqlite::Connection as AsyncConnection;

pub struct CacheStore {
    db: Arc<AsyncConnection>,
}

impl CacheStore {
    pub async fn new(path: &str) -> Result<Self, TesseraError> {
        let db = Arc::new(
            AsyncConnection::open(path)
                .await
                .map_err(|e| TesseraError::DatabaseError(e))?,
        );

        let db_clone = db.clone();
        db_clone
            .call(move |conn| {
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS adapters (
                        adapter_id TEXT PRIMARY KEY,
                        adapter_path TEXT NOT NULL,
                        base_model TEXT NOT NULL,
                        rank INTEGER NOT NULL,
                        source_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        quality_score REAL DEFAULT 0.0
                    )",
                    [],
                )?;
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_base_model ON adapters(base_model)",
                    [],
                )?;
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_source_type ON adapters(source_type)",
                    [],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| TesseraError::DatabaseError(e))?;

        Ok(CacheStore { db })
    }

    pub async fn record_adapter(
        &self,
        adapter_id: &str,
        adapter_path: &str,
        base_model: &str,
        rank: u32,
        source_type: &str,
    ) -> Result<(), TesseraError> {
        let db = self.db.clone();
        let adapter_id = adapter_id.to_string();
        let adapter_path = adapter_path.to_string();
        let base_model = base_model.to_string();
        let source_type = source_type.to_string();
        let created_at = chrono::Utc::now().to_rfc3339();

        db.call(move |conn| {
            conn.execute(
                "INSERT OR REPLACE INTO adapters 
                 (adapter_id, adapter_path, base_model, rank, source_type, created_at, hit_count, quality_score)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, 0.8)",
                params![adapter_id, adapter_path, base_model, rank, source_type, created_at],
            )?;
            Ok(())
        })
        .await
        .map_err(|e| TesseraError::DatabaseError(e))
    }

    pub async fn increment_hit_count(&self, adapter_id: &str) -> Result<(), TesseraError> {
        let db = self.db.clone();
        let adapter_id = adapter_id.to_string();

        db.call(move |conn| {
            conn.execute(
                "UPDATE adapters SET hit_count = hit_count + 1 WHERE adapter_id = ?1",
                params![adapter_id],
            )?;
            Ok(())
        })
        .await
        .map_err(|e| TesseraError::DatabaseError(e))
    }

    pub async fn get_adapter_metadata(&self, adapter_id: &str) -> Result<Option<AdapterMetadataRow>, TesseraError> {
        let db = self.db.clone();
        let adapter_id = adapter_id.to_string();

        db.call(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT adapter_id, adapter_path, base_model, rank, hit_count, created_at 
                 FROM adapters WHERE adapter_id = ?1",
            )?;
            let result = stmt.query_row(params![adapter_id], |row| {
                Ok(AdapterMetadataRow {
                    adapter_id: row.get(0)?,
                    adapter_path: row.get(1)?,
                    base_model: row.get(2)?,
                    rank: row.get::<_, i64>(3)? as u32,
                    hit_count: row.get::<_, i64>(4)? as u64,
                    created_at: row.get(5)?,
                })
            });
            match result {
                Ok(meta) => Ok(Some(meta)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e),
            }
        })
        .await
        .map_err(|e| TesseraError::DatabaseError(e))
    }

    pub async fn get_adapter_path(&self, adapter_id: &str) -> Result<Option<String>, TesseraError> {
        let db = self.db.clone();
        let adapter_id = adapter_id.to_string();

        db.call(move |conn| {
            let mut stmt = conn.prepare("SELECT adapter_path FROM adapters WHERE adapter_id = ?1")?;
            let result = stmt.query_row(params![adapter_id], |row| row.get(0));
            match result {
                Ok(path) => Ok(Some(path)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e),
            }
        })
        .await
        .map_err(|e| TesseraError::DatabaseError(e))
    }

    pub async fn get_stats(&self) -> Result<CacheStats, TesseraError> {
        let db = self.db.clone();

        db.call(move |conn| {
            let total: u64 = conn.query_row("SELECT COUNT(*) FROM adapters", [], |row| row.get(0))?;
            let hits: u64 = conn.query_row("SELECT SUM(hit_count) FROM adapters", [], |row| row.get(0)).unwrap_or(0);
            let avg_quality: f64 = conn.query_row("SELECT AVG(quality_score) FROM adapters", [], |row| row.get(0)).unwrap_or(0.0);
            
            Ok(CacheStats {
                total_adapters: total,
                total_hits: hits,
                avg_quality,
            })
        })
        .await
        .map_err(|e| TesseraError::DatabaseError(e))
    }
}

#[derive(Debug, Default)]
pub struct CacheStats {
    pub total_adapters: u64,
    pub total_hits: u64,
    pub avg_quality: f64,
}

#[derive(Debug)]
pub struct AdapterMetadataRow {
    pub adapter_id: String,
    pub adapter_path: String,
    pub base_model: String,
    pub rank: u32,
    pub hit_count: u64,
    pub created_at: String,
}
