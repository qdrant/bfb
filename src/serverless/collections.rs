//! Track which serverless collections existed before a run and which were
//! created during upload, creating missing ones lazily on first upsert.

use std::collections::HashSet;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use qdrant_client::serverless::{CollectionConfig, CollectionSummary, QdrantServerless};
use tokio::sync::Mutex as AsyncMutex;

/// Name of the `i`-th collection under `prefix` (`benchmark-` + `0` → `benchmark-0`).
pub fn collection_name(prefix: &str, index: usize) -> String {
    format!("{prefix}{index}")
}

/// All collections currently in the space whose name starts with `prefix`,
/// sorted by name.
pub async fn list_matching(
    client: &QdrantServerless,
    prefix: &str,
) -> Result<Vec<CollectionSummary>> {
    let mut summaries: Vec<CollectionSummary> = client
        .list_collections()
        .await
        .context("list_collections")?
        .into_iter()
        .filter(|c| c.collection_name.starts_with(prefix))
        .collect();
    summaries.sort_by(|a, b| a.collection_name.cmp(&b.collection_name));
    Ok(summaries)
}

/// Check that every vector the upload config will send exists in `existing`
/// with the same shape. Extra vectors or indexes on the existing collection
/// are fine; a missing or mismatched one would make every upsert fail, so
/// report it up front instead.
pub fn check_vector_shape(
    name: &str,
    existing: &CollectionConfig,
    wanted: &CollectionConfig,
) -> Result<()> {
    for (vector, want) in &wanted.dense_vectors {
        let label = if vector.is_empty() {
            "default dense vector".to_string()
        } else {
            format!("dense vector {vector:?}")
        };
        let Some(have) = existing.dense_vectors.get(vector) else {
            bail!(
                "collection {name} has no {label}; delete it (`bfb serverless clear`) or use a matching config"
            );
        };
        if have.size != want.size || have.distance != want.distance {
            bail!(
                "collection {name}: {label} is {}d/{:?} but the config wants {}d/{:?}",
                have.size,
                have.distance,
                want.size,
                want.distance
            );
        }
        if have.multivector != want.multivector {
            bail!(
                "collection {name}: {label} multivector={} but the config wants {}",
                have.multivector,
                want.multivector
            );
        }
    }
    for vector in wanted.sparse_vectors.keys() {
        if !existing.sparse_vectors.contains_key(vector) {
            bail!("collection {name} has no sparse vector {vector:?}");
        }
    }
    Ok(())
}

/// Registry for one serverless upload experiment.
pub struct CollectionRegistry {
    prefix: String,
    /// Present in the space before the experiment started.
    preexisting: HashSet<String>,
    /// Known to exist and match the config: created by this process, or
    /// preexisting and verified on first use.
    ready: Mutex<HashSet<String>>,
    /// Created by this process during upload.
    created: Mutex<HashSet<String>>,
    /// Config used when lazily creating a missing collection.
    create_config: CollectionConfig,
    /// One lock per slot, so parallel workers on the same collection do not
    /// race `create_collection` while other slots proceed unblocked.
    slot_locks: Vec<AsyncMutex<()>>,
}

impl CollectionRegistry {
    pub async fn bootstrap(
        client: &QdrantServerless,
        prefix: &str,
        collections_count: usize,
        create_config: CollectionConfig,
    ) -> Result<Self> {
        let preexisting: HashSet<String> = list_matching(client, prefix)
            .await?
            .into_iter()
            .map(|c| c.collection_name)
            .collect();

        println!(
            "Serverless registry: prefix={prefix:?} slots={collections_count} preexisting={}",
            preexisting.len()
        );
        let mut shown: Vec<&String> = preexisting.iter().collect();
        shown.sort();
        for name in shown.iter().take(5) {
            println!("  preexisting: {name}");
        }
        if preexisting.len() > 5 {
            println!("  … {} more", preexisting.len() - 5);
        }

        Ok(Self {
            prefix: prefix.to_string(),
            preexisting,
            ready: Mutex::new(HashSet::new()),
            created: Mutex::new(HashSet::new()),
            create_config,
            slot_locks: (0..collections_count)
                .map(|_| AsyncMutex::new(()))
                .collect(),
        })
    }

    pub fn name(&self, index: usize) -> String {
        collection_name(&self.prefix, index)
    }

    /// Ensure collection `index` exists with a compatible vector shape,
    /// creating it lazily on first use. Returns the collection name.
    pub async fn ensure(&self, client: &QdrantServerless, index: usize) -> Result<String> {
        let name = self.name(index);

        if self.ready.lock().unwrap().contains(&name) {
            return Ok(name);
        }

        let _guard = self.slot_locks[index].lock().await;

        // Re-check under the slot lock.
        if self.ready.lock().unwrap().contains(&name) {
            return Ok(name);
        }

        let info = client
            .get_collection(&name)
            .await
            .with_context(|| format!("get_collection {name}"))?;

        if info.exists {
            // Preexisting, or appeared between bootstrap and now (another
            // process / prior run). Do not claim we created it, but make sure
            // the upload config fits it before sending points.
            if let Some(existing) = &info.config {
                check_vector_shape(&name, existing, &self.create_config)?;
            }
            if !self.preexisting.contains(&name) {
                println!("Collection {name} appeared during the run; reusing it");
            }
        } else {
            client
                .create_collection(&name, self.create_config.clone())
                .await
                .with_context(|| format!("create_collection {name}"))?;
            self.created.lock().unwrap().insert(name.clone());
            println!("Created collection {name}");
        }

        self.ready.lock().unwrap().insert(name.clone());
        Ok(name)
    }

    pub fn summary(&self) {
        let created = self.created.lock().unwrap();
        let ready = self.ready.lock().unwrap();
        println!(
            "Serverless registry summary: preexisting={} created={} used={}",
            self.preexisting.len(),
            created.len(),
            ready.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::serverless::{DenseVectorConfig, Distance, SparseVectorConfig};

    fn dense(size: u64) -> CollectionConfig {
        CollectionConfig::new().dense_vector(DenseVectorConfig::new(size, Distance::Cosine))
    }

    #[test]
    fn matching_shape_passes_and_extra_vectors_are_fine() {
        let existing = dense(128).named_sparse_vector("text", SparseVectorConfig::new());
        check_vector_shape("c", &existing, &dense(128)).unwrap();
    }

    #[test]
    fn size_mismatch_is_reported() {
        let err = check_vector_shape("c", &dense(128), &dense(256)).unwrap_err();
        assert!(err.to_string().contains("128d"), "{err}");
    }

    #[test]
    fn missing_sparse_vector_is_reported() {
        let wanted = dense(128).named_sparse_vector("text", SparseVectorConfig::new());
        let err = check_vector_shape("c", &dense(128), &wanted).unwrap_err();
        assert!(err.to_string().contains("sparse vector \"text\""), "{err}");
    }
}
