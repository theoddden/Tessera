use crate::error::TesseraError;
use safetensors::SafeTensors;
use std::path::PathBuf;

pub struct AdapterStore {
    base_path: PathBuf,
}

impl AdapterStore {
    pub fn new(base_path: &str) -> Result<Self, TesseraError> {
        let path = PathBuf::from(base_path);

        if !path.exists() {
            std::fs::create_dir_all(&path).map_err(|e| TesseraError::IoError(e))?;
        }

        Ok(AdapterStore { base_path: path })
    }

    pub async fn save(&self, adapter_id: &str, bytes: &[u8]) -> Result<PathBuf, TesseraError> {
        // Validate safetensors format before storing
        SafeTensors::deserialize(bytes).map_err(|e| TesseraError::InvalidAdapter(e.to_string()))?;

        let path = self.base_path.join(format!("{}.safetensors", adapter_id));

        // Async write — tokio fs
        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| TesseraError::IoError(e))?;

        Ok(path)
    }

    pub async fn load(&self, adapter_path: &str) -> Result<Vec<u8>, TesseraError> {
        // Zero-copy read where possible
        let bytes = tokio::fs::read(adapter_path)
            .await
            .map_err(|e| TesseraError::IoError(e))?;

        // Validate on load
        SafeTensors::deserialize(&bytes)
            .map_err(|e| TesseraError::CorruptAdapter(e.to_string()))?;

        Ok(bytes)
    }

    pub async fn load_by_id(&self, adapter_id: &str) -> Result<Vec<u8>, TesseraError> {
        let path = self.base_path.join(format!("{}.safetensors", adapter_id));
        self.load(path.to_str().unwrap()).await
    }

    pub async fn delete(&self, adapter_id: &str) -> Result<(), TesseraError> {
        let path = self.base_path.join(format!("{}.safetensors", adapter_id));

        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| TesseraError::IoError(e))?;
        }

        Ok(())
    }

    pub async fn list(&self) -> Result<Vec<String>, TesseraError> {
        let mut entries = tokio::fs::read_dir(&self.base_path)
            .await
            .map_err(|e| TesseraError::IoError(e))?;

        let mut adapter_ids = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| TesseraError::IoError(e))?
        {
            let path = entry.path();
            if path
                .extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
            {
                if let Some(stem) = path.file_stem() {
                    if let Some(id) = stem.to_str() {
                        adapter_ids.push(id.to_string());
                    }
                }
            }
        }

        Ok(adapter_ids)
    }
}
