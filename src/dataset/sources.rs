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
    /// Whole-payload source (`collection.payload.source`).
    payload_object: Option<DatasetReader>,
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
            .fields
            .iter()
            .map(|p| {
                let Some(source) = &p.source else {
                    return Ok(None);
                };
                if source.kind == PayloadSourceKind::Dataset {
                    let dataset = source
                        .dataset
                        .as_ref()
                        .context("payload dataset source is missing dataset fields")?;
                    let field = source
                        .field
                        .clone()
                        .context("payload dataset source is missing `field`")?;
                    Ok(Some((DatasetReader::open(datasets_dir, dataset)?, field)))
                } else {
                    Ok(None)
                }
            })
            .collect::<anyhow::Result<_>>()?;

        // Whole-payload source (`payload.source`).
        let payload_object = match &config.collection.payload.source {
            Some(source) if source.kind == PayloadSourceKind::Dataset => {
                let dataset = source
                    .dataset
                    .as_ref()
                    .context("collection `source` is missing dataset fields")?;
                Some(DatasetReader::open(datasets_dir, dataset)?)
            }
            _ => None,
        };

        Ok(Self {
            vector,
            sparse,
            payload,
            payload_object,
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

    /// A point's sub-vectors from a `multivector` dataset (ColBERT-style).
    pub fn multi_dense_vector(&self, slot: usize, idx: u64) -> Option<Vec<Vec<f32>>> {
        let reader = self.vector.get(slot)?.as_ref()?;
        Some(
            reader
                .multi_dense_vector(idx as usize)
                .unwrap_or_else(|e| panic!("failed to read multivector at {idx}: {e}")),
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

    /// Whole payload object for a point, from the collection-level source.
    /// Returns the object's fields (values passed through unchanged).
    pub fn payload_object(&self, idx: u64) -> Option<Vec<(String, Value)>> {
        let reader = self.payload_object.as_ref()?;
        let value = reader
            .payload_object(idx as usize)
            .unwrap_or_else(|e| panic!("failed to read payload object at {idx}: {e}"))
            .unwrap_or_else(|| panic!("payload object missing at index {idx} in dataset"));
        match value {
            Value::Object(map) => Some(
                map.into_iter()
                    .map(|(k, v)| (k, json_to_payload_value(v)))
                    .collect(),
            ),
            other => panic!("payload object at {idx} is not a JSON object: {other}"),
        }
    }
}
