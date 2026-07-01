use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use ndarray::{ArrayView2, Axis};
use ndarray_npy::ViewNpyExt;
use serde_json::Value;

use super::jsonl::JsonlStore;

pub struct TarReader {
    vectors: ArrayView2<'static, f32>,
    _vector_mmap: Mmap,
    payloads: Option<JsonlStore>,
}

impl TarReader {
    pub fn open(path: &Path) -> Result<Self> {
        let vectors_path = path.join("vectors.npy");
        let file = File::open(&vectors_path)
            .with_context(|| format!("failed to open {}", vectors_path.display()))?;
        let mmap = unsafe { Mmap::map(&file).context("failed to mmap vectors.npy")? };
        let vectors = ArrayView2::<f32>::view_npy(&mmap)
            .with_context(|| format!("failed to parse {}", vectors_path.display()))?;
        // SAFETY: mmap lives in `_vector_mmap` on the same struct.
        let vectors = unsafe {
            std::mem::transmute::<ArrayView2<'_, f32>, ArrayView2<'static, f32>>(vectors)
        };

        let payloads_path = path.join("payloads.jsonl");
        let payloads = if payloads_path.exists() {
            Some(JsonlStore::open(&payloads_path)?)
        } else {
            None
        };

        Ok(TarReader {
            vectors,
            _vector_mmap: mmap,
            payloads,
        })
    }

    pub fn num_points(&self) -> usize {
        self.vectors.len_of(Axis(0))
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.num_points() {
            bail!(
                "index {idx} out of range (dataset has {} points)",
                self.num_points()
            );
        }
        Ok(self.vectors.row(idx).iter().copied().collect())
    }

    pub fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        let Some(store) = &self.payloads else {
            return Ok(None);
        };
        let line = store
            .value_at(idx)?
            .with_context(|| format!("payload index {idx} out of range"))?;
        Ok(line.get(field).cloned())
    }
}
