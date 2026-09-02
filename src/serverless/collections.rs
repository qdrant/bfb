//! Track which serverless collections existed before a run, which were created
//! during upload, and which can be queried afterwards.

use std::collections::HashSet;
use std::sync::Mutex;

use anyhow::{Context, Result};
use qdrant_client::serverless::{CollectionConfig, QdrantServerless};
use tokio::sync::Mutex as AsyncMutex;

/// Name of the `i`-th collection under `prefix` (`benchmark-` + `0` → `benchmark-0`).
pub fn collection_name(prefix: &str, index: usize) -> String {
    format!("{prefix}{index}")
}

/// All collections currently in the space whose name starts with `prefix`.
pub async fn list_matching(client: &QdrantServerless, prefix: &str) -> Result<Vec<String>> {
    let mut names: Vec<String> = client
        .list_collections()
        .await
        .context("list_collections")?
        .into_iter()
        .map(|c| c.collection_name)
        .filter(|n| n.starts_with(prefix))
        .collect();
    names.sort();
    Ok(names)
}

/// Registry for one serverless upload/query experiment.
pub struct CollectionRegistry {
    prefix: String,
    /// Present in the space before the experiment started.
    preexisting: HashSet<String>,
    /// Created by this process during upload.
    created: Mutex<HashSet<String>>,
    /// Have received at least one successful upsert (or existed with points).
    queryable: Mutex<HashSet<String>>,
    /// Config used when lazily creating a missing collection.
    create_config: CollectionConfig,
    /// Serializes create RPCs so parallel workers do not race on the same name.
    create_lock: AsyncMutex<()>,
}

impl CollectionRegistry {
    pub async fn bootstrap(
        client: &QdrantServerless,
        prefix: &str,
        collections_count: usize,
        create_config: CollectionConfig,
    ) -> Result<Self> {
        let preexisting: HashSet<String> =
            list_matching(client, prefix).await?.into_iter().collect();

        println!(
            "Serverless registry: prefix={prefix:?} slots={collections_count} preexisting={}",
            preexisting.len()
        );
        for name in preexisting.iter().take(5) {
            println!("  preexisting: {name}");
        }
        if preexisting.len() > 5 {
            println!("  … {} more", preexisting.len() - 5);
        }

        Ok(Self {
            prefix: prefix.to_string(),
            preexisting,
            created: Mutex::new(HashSet::new()),
            queryable: Mutex::new(HashSet::new()),
            create_config,
            create_lock: AsyncMutex::new(()),
        })
    }

    pub fn name(&self, index: usize) -> String {
        collection_name(&self.prefix, index)
    }

    /// Ensure collection `index` exists, creating it lazily on first use.
    /// Returns the collection name.
    pub async fn ensure(&self, client: &QdrantServerless, index: usize) -> Result<String> {
        let name = self.name(index);

        // Fast path: already known to exist (preexisting or created this run).
        {
            let created = self.created.lock().unwrap();
            if self.preexisting.contains(&name) || created.contains(&name) {
                return Ok(name);
            }
        }

        // Slow path: serialize creates so parallel upserts of the same slot
        // do not all race `create_collection`.
        let _guard = self.create_lock.lock().await;

        // Re-check under the lock.
        {
            let created = self.created.lock().unwrap();
            if self.preexisting.contains(&name) || created.contains(&name) {
                return Ok(name);
            }
        }

        let info = client
            .get_collection(&name)
            .await
            .with_context(|| format!("get_collection {name}"))?;

        if info.exists {
            // Appeared between bootstrap and now (another process / prior run).
            // Do not claim we created it.
            return Ok(name);
        }

        client
            .create_collection(&name, self.create_config.clone())
            .await
            .with_context(|| format!("create_collection {name}"))?;
        self.created.lock().unwrap().insert(name.clone());
        println!("Created collection {name}");
        Ok(name)
    }

    pub fn mark_queryable(&self, name: &str) {
        self.queryable.lock().unwrap().insert(name.to_string());
    }

    pub fn summary(&self) {
        let created = self.created.lock().unwrap();
        let queryable = self.queryable.lock().unwrap();
        println!(
            "Serverless registry summary: preexisting={} created={} queryable={}",
            self.preexisting.len(),
            created.len(),
            queryable.len()
        );
    }
}
