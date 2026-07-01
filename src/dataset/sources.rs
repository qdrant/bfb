//! Open dataset readers aligned with an upload config.

use std::path::Path;

use anyhow::Context;
use serde_json::Value;

use crate::config::{PayloadSourceKind, SparseKind, UploadConfig, VectorSource};

use super::payload::json_to_payload_value;
use super::reader::DatasetReader;

/// Dataset readers opened for each vector / sparse / payload slot in the config.
pub struct UploadDatasetSources {
    vector: Vec<Option<DatasetReader>>,
    sparse: Vec<Option<DatasetReader>>,
    payload: Vec<Option<(DatasetReader, String)>>,
}

impl UploadDatasetSources {
    pub fn open(config: &UploadConfig, datasets_dir: &Path) -> anyhow::Result<Self> {
        let vector = config
            .collection
            .vectors
            .iter()
            .map(|v| match &v.source {
                VectorSource::Dataset { dataset } => {
                    DatasetReader::open(datasets_dir, dataset).map(Some)
                }
                _ => Ok(None),
            })
            .collect::<anyhow::Result<_>>()?;

        let sparse = config
            .collection
            .sparse_vectors
            .iter()
            .map(|s| {
                if s.source.kind == SparseKind::Dataset {
                    let dataset = s
                        .source
                        .dataset
                        .as_ref()
                        .context("sparse dataset source is missing dataset fields")?;
                    DatasetReader::open(datasets_dir, dataset).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<anyhow::Result<_>>()?;

        let payload = config
            .collection
            .payloads
            .iter()
            .map(|p| {
                if p.source.kind == PayloadSourceKind::Dataset {
                    let dataset = p
                        .source
                        .dataset
                        .as_ref()
                        .context("payload dataset source is missing dataset fields")?;
                    let field = p
                        .source
                        .field
                        .clone()
                        .context("payload dataset source is missing `field`")?;
                    Ok(Some((DatasetReader::open(datasets_dir, dataset)?, field)))
                } else {
                    Ok(None)
                }
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(Self {
            vector,
            sparse,
            payload,
        })
    }

    pub fn dense_vector(&self, slot: usize, idx: u64) -> Option<Vec<f32>> {
        let reader = self.vector.get(slot)?.as_ref()?;
        Some(
            reader
                .dense_vector(idx as usize)
                .unwrap_or_else(|e| panic!("failed to read dataset vector at {idx}: {e}")),
        )
    }

    pub fn sparse_vector(&self, slot: usize, idx: u64) -> Option<Vec<(u32, f32)>> {
        let reader = self.sparse.get(slot)?.as_ref()?;
        Some(
            reader
                .sparse_vector(idx as usize)
                .unwrap_or_else(|e| panic!("failed to read sparse dataset vector at {idx}: {e}")),
        )
    }

    pub fn payload_value(&self, slot: usize, idx: u64) -> Option<Value> {
        let (reader, field) = self.payload.get(slot)?.as_ref()?;
        let value = reader
            .payload_field(idx as usize, field)
            .unwrap_or_else(|e| panic!("failed to read payload field {field} at {idx}: {e}"))
            .unwrap_or_else(|| panic!("payload field {field} missing at index {idx} in dataset"));
        Some(json_to_payload_value(value))
    }
}
