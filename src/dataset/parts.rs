//! Sharded datasets: a numbered family of files presented as one row space.
//!
//! Corpora too large to publish as a single artifact ship as parts —
//! LAION-400M is 410 `img_emb_{i}.npy` / `metadata_{i}.parquet` pairs. A
//! `parts:` block turns those into one logical dataset, so point ids stay
//! global and `--offset` resumes an interrupted run at the right row.
//!
//! # Sizing
//!
//! Mapping a global row to a part needs every part's row count up front, and
//! the counts are *not* uniform: LAION's parts are 1,000,448 rows except part
//! 408 (1,000,501) and the last (518,720). A configured "rows per part" would
//! therefore be wrong for the tail of the corpus, silently misaligning payloads
//! against vectors, so the counts are always measured instead.
//!
//! Measuring is cheap because both formats keep their shape at a known end of
//! the file: the `.npy` header is the first ~128 bytes, and the parquet footer
//! the last few KB. One ranged request per part sizes the whole corpus without
//! downloading any of it, and the result is cached in a sidecar so later runs
//! do no requests at all.

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::config::{DatasetKind, PartsConfig, ResolvedDatasetConfig};
use super::download::fetch_range;
use super::readers::{NpyReader, ParquetReader, parquet_row_count, parse_npy_header};

/// Parts kept open at once. Upload walks rows in order, so two is enough to
/// straddle a boundary; holding more would pin every part's decode buffers.
const OPEN_PARTS: usize = 2;

/// Prefix of a `.npy` requested when sizing a remote part. Far more than the
/// ~128 bytes a 2-D header occupies, and still a single small request.
const NPY_HEADER_PROBE: usize = 4096;

/// Tail of a parquet file requested when sizing a remote part. Comfortably
/// covers a typical footer (LAION's are ~9.5 KB) in one request; a larger
/// footer costs one more.
const PARQUET_TAIL_PROBE: usize = 64 * 1024;

/// Concurrent sizing requests. Enough to hide per-request latency across a few
/// hundred parts without hammering the host.
const PROBE_CONCURRENCY: usize = 16;

/// One part's measured size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartEntry {
    /// The part's own number, as substituted into the path/link templates.
    pub index: usize,
    pub rows: usize,
    pub bytes: u64,
}

/// Cached sizing result, invalidated by any change to the parts spec.
#[derive(Debug, Serialize, Deserialize)]
struct PartsManifest {
    key: String,
    parts: Vec<PartEntry>,
}

/// Everything needed to locate, fetch and open one part.
pub struct PartSource {
    datasets_dir: PathBuf,
    name: String,
    kind: DatasetKind,
    parts: PartsConfig,
    columns: Option<Vec<String>>,
    exclude: Vec<String>,
    fill_null: Option<Value>,
}

impl PartSource {
    pub fn new(datasets_dir: &Path, config: &ResolvedDatasetConfig, parts: PartsConfig) -> Self {
        PartSource {
            datasets_dir: datasets_dir.to_path_buf(),
            name: config.name.clone(),
            kind: config.kind,
            parts,
            columns: config.columns.clone(),
            exclude: config.exclude.clone(),
            fill_null: config.fill_null.clone(),
        }
    }

    fn local_path(&self, index: usize) -> PathBuf {
        self.datasets_dir.join(expand(&self.parts.path, index))
    }

    fn link(&self, index: usize) -> Option<String> {
        self.parts.link.as_deref().map(|tpl| expand(tpl, index))
    }

    /// Identity of the parts spec: a different spec must not reuse a manifest.
    fn key(&self) -> String {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.kind).hash(&mut hasher);
        self.parts.path.hash(&mut hasher);
        self.parts.link.hash(&mut hasher);
        self.parts.start.hash(&mut hasher);
        self.parts.count.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn manifest_path(&self) -> PathBuf {
        self.datasets_dir
            .join(".parts-index")
            .join(format!("{}.json", sanitize(&self.name)))
    }

    fn indices(&self) -> impl Iterator<Item = usize> + use<> {
        let start = self.parts.start;
        start..start + self.parts.count
    }

    /// Ensure part `index` is present locally, downloading it if needed.
    pub fn ensure_downloaded(&self, index: usize) -> Result<PathBuf> {
        let target = self.local_path(index);
        if target.exists() {
            return Ok(target);
        }
        let link = self.link(index).with_context(|| {
            format!(
                "dataset {:?} part {index} is missing at {} and no `parts.link` is configured",
                self.name,
                target.display()
            )
        })?;
        super::download::download_file_to(&link, &target)?;
        Ok(target)
    }

    fn open_reader(&self, path: &Path) -> Result<PartReader> {
        Ok(match self.kind {
            DatasetKind::Npy => PartReader::Npy(NpyReader::open(path)?),
            DatasetKind::Parquet => PartReader::Parquet(Box::new(ParquetReader::open(
                path,
                self.columns.as_deref(),
                &self.exclude,
                self.fill_null.as_ref(),
            )?)),
            other => bail!("`parts:` is not supported for format {other:?} (use npy or parquet)"),
        })
    }

    /// Measure one part's row count, without downloading it if it is remote.
    fn probe(&self, index: usize) -> Result<PartEntry> {
        let local = self.local_path(index);
        let (rows, bytes) = if local.exists() {
            let bytes = std::fs::metadata(&local)
                .with_context(|| format!("failed to stat {}", local.display()))?
                .len();
            (measure_local(self.kind, &local)?, bytes)
        } else {
            let link = self.link(index).with_context(|| {
                format!(
                    "dataset {:?} part {index} is missing at {} and no `parts.link` is configured",
                    self.name,
                    local.display()
                )
            })?;
            measure_remote(self.kind, &link)?
        };
        Ok(PartEntry { index, rows, bytes })
    }

    /// Size every part, using the cached manifest when the spec is unchanged.
    fn measure_all(&self) -> Result<Vec<PartEntry>> {
        let key = self.key();
        let manifest_path = self.manifest_path();
        if let Some(cached) = read_manifest(&manifest_path, &key) {
            return Ok(cached);
        }

        println!(
            "Sizing {} parts of dataset {:?}...",
            self.parts.count, self.name
        );
        let indices: Vec<usize> = self.indices().collect();
        let results: Mutex<Vec<Option<Result<PartEntry>>>> =
            Mutex::new((0..indices.len()).map(|_| None).collect());

        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..PROBE_CONCURRENCY.min(indices.len()) {
                scope.spawn(|| {
                    loop {
                        let slot = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&index) = indices.get(slot) else {
                            break;
                        };
                        let probed = self.probe(index);
                        results.lock().unwrap()[slot] = Some(probed);
                    }
                });
            }
        });

        let entries = results
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|slot| slot.expect("every slot is filled before the scope ends"))
            .collect::<Result<Vec<_>>>()?;

        let total: usize = entries.iter().map(|e| e.rows).sum();
        println!(
            "Dataset {:?}: {} parts, {total} rows total",
            self.name, self.parts.count
        );
        write_manifest(&manifest_path, &key, &entries);
        Ok(entries)
    }
}

/// Substitute a part number into a `{i}` template.
fn expand(template: &str, index: usize) -> String {
    template.replace("{i}", &index.to_string())
}

/// Keep a dataset name usable as a file name.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn measure_local(kind: DatasetKind, path: &Path) -> Result<usize> {
    match kind {
        DatasetKind::Npy => {
            use std::io::Read;
            let mut file = std::fs::File::open(path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            let mut head = vec![0u8; NPY_HEADER_PROBE];
            let read = file
                .read(&mut head)
                .with_context(|| format!("failed to read {}", path.display()))?;
            head.truncate(read);
            Ok(parse_npy_header(&head)
                .with_context(|| format!("failed to parse {}", path.display()))?
                .num_points)
        }
        DatasetKind::Parquet => parquet_row_count(path),
        other => bail!("`parts:` is not supported for format {other:?} (use npy or parquet)"),
    }
}

fn measure_remote(kind: DatasetKind, link: &str) -> Result<(usize, u64)> {
    match kind {
        DatasetKind::Npy => {
            let response = fetch_range(link, &format!("bytes=0-{}", NPY_HEADER_PROBE - 1))?;
            let layout = parse_npy_header(&response.body)
                .with_context(|| format!("failed to parse the .npy header of {link}"))?;
            Ok((layout.num_points, response.total_len))
        }
        DatasetKind::Parquet => {
            let mut response = fetch_range(link, &format!("bytes=-{PARQUET_TAIL_PROBE}"))?;
            // The footer length lives in the last 8 bytes; if the footer runs
            // past what we fetched, ask for exactly as much as it needs.
            if let Some(needed) = super::readers::parquet_footer_len(&response.body)?
                && needed > response.body.len()
            {
                response = fetch_range(link, &format!("bytes=-{needed}"))?;
            }
            let rows =
                super::readers::parquet_row_count_from_tail(&response.body, response.total_len)
                    .with_context(|| format!("failed to parse the parquet footer of {link}"))?;
            Ok((rows, response.total_len))
        }
        other => bail!("`parts:` is not supported for format {other:?} (use npy or parquet)"),
    }
}

fn read_manifest(path: &Path, key: &str) -> Option<Vec<PartEntry>> {
    let text = std::fs::read_to_string(path).ok()?;
    let manifest: PartsManifest = serde_json::from_str(&text).ok()?;
    (manifest.key == key).then_some(manifest.parts)
}

/// Best-effort: an unwritable cache costs a re-probe next run, nothing more.
fn write_manifest(path: &Path, key: &str, parts: &[PartEntry]) {
    let manifest = PartsManifest {
        key: key.to_string(),
        parts: parts.to_vec(),
    };
    let Ok(text) = serde_json::to_string(&manifest) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, text);
}

/// One open part. The parquet reader carries a decode ring, so it is boxed to
/// keep the enum from sizing every `npy` part by it too.
pub enum PartReader {
    Npy(NpyReader),
    Parquet(Box<ParquetReader>),
}

impl PartReader {
    fn num_points(&self) -> usize {
        match self {
            PartReader::Npy(r) => r.num_points(),
            PartReader::Parquet(r) => r.num_points(),
        }
    }

    fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        match self {
            PartReader::Npy(r) => r.vector_at(idx),
            PartReader::Parquet(_) => bail!("parquet parts do not contain dense vectors"),
        }
    }

    fn payload_object(&self, idx: usize) -> Result<Option<Value>> {
        match self {
            PartReader::Parquet(r) => r.payload_object(idx),
            PartReader::Npy(_) => bail!("npy parts do not contain payloads"),
        }
    }

    fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        match self {
            PartReader::Parquet(r) => r.payload_field(idx, field),
            PartReader::Npy(_) => bail!("npy parts do not contain payloads"),
        }
    }
}

/// A family of parts addressed as one contiguous row space.
pub struct PartitionedReader {
    source: PartSource,
    entries: Vec<PartEntry>,
    /// Global index of each part's first row, plus a final total.
    starts: Vec<usize>,
    open: RwLock<VecDeque<(usize, std::sync::Arc<PartReader>)>>,
    /// Serializes the open-a-new-part path so concurrent readers crossing a
    /// boundary fetch it once. Held *outside* `open`, so readers still working
    /// on the previous part are never blocked behind a download.
    opening: Mutex<()>,
}

impl PartitionedReader {
    pub fn open(datasets_dir: &Path, config: &ResolvedDatasetConfig) -> Result<Self> {
        let parts = config
            .parts
            .clone()
            .expect("PartitionedReader requires a `parts:` block");
        let source = PartSource::new(datasets_dir, config, parts);
        let entries = source.measure_all()?;

        let mut starts = Vec::with_capacity(entries.len() + 1);
        let mut running = 0usize;
        for entry in &entries {
            starts.push(running);
            running += entry.rows;
        }
        starts.push(running);

        Ok(PartitionedReader {
            source,
            entries,
            starts,
            open: RwLock::new(VecDeque::new()),
            opening: Mutex::new(()),
        })
    }

    pub fn num_points(&self) -> usize {
        *self.starts.last().unwrap_or(&0)
    }

    /// Split a global row index into (slot in `entries`, row within the part).
    fn locate(&self, idx: usize) -> Result<(usize, usize)> {
        if idx >= self.num_points() {
            bail!(
                "row {idx} is past the end of dataset {:?} ({} rows)",
                self.source.name,
                self.num_points()
            );
        }
        let slot = match self.starts.binary_search(&idx) {
            Ok(exact) => exact,
            Err(next) => next - 1,
        };
        Ok((slot, idx - self.starts[slot]))
    }

    fn reader_for(&self, slot: usize) -> Result<std::sync::Arc<PartReader>> {
        let index = self.entries[slot].index;
        if let Some((_, reader)) = self
            .open
            .read()
            .unwrap()
            .iter()
            .find(|(open, _)| *open == index)
        {
            return Ok(reader.clone());
        }

        let _guard = self.opening.lock().unwrap();
        // Another thread may have opened it while we waited for the guard.
        if let Some((_, reader)) = self
            .open
            .read()
            .unwrap()
            .iter()
            .find(|(open, _)| *open == index)
        {
            return Ok(reader.clone());
        }

        let path = self.source.ensure_downloaded(index)?;
        let reader = self.source.open_reader(&path)?;

        // A stale sidecar would misalign every row after this part; catching it
        // here costs nothing, since opening already read the real count.
        let expected = self.entries[slot].rows;
        if reader.num_points() != expected {
            bail!(
                "{} holds {} rows but the cached parts index says {expected}; \
                 delete {} and re-run to re-measure",
                path.display(),
                reader.num_points(),
                self.source.manifest_path().display()
            );
        }

        let reader = std::sync::Arc::new(reader);
        let mut open = self.open.write().unwrap();
        open.push_back((index, reader.clone()));
        while open.len() > OPEN_PARTS {
            open.pop_front();
        }
        Ok(reader)
    }

    pub fn vector_at(&self, idx: usize) -> Result<Vec<f32>> {
        let (slot, local) = self.locate(idx)?;
        self.reader_for(slot)?.vector_at(local)
    }

    pub fn payload_object(&self, idx: usize) -> Result<Option<Value>> {
        let (slot, local) = self.locate(idx)?;
        self.reader_for(slot)?.payload_object(local)
    }

    pub fn payload_field(&self, idx: usize, field: &str) -> Result<Option<Value>> {
        let (slot, local) = self.locate(idx)?;
        self.reader_for(slot)?.payload_field(local, field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::config::DatasetConfig;
    use crate::dataset::fixtures::{make_ramp_npy, write_parquet};

    fn resolved(dir: &Path, kind: DatasetKind, path: &str, count: usize) -> ResolvedDatasetConfig {
        let _ = dir;
        let config = DatasetConfig {
            name: "test-parts".to_string(),
            kind: Some(kind),
            parts: Some(PartsConfig {
                count,
                start: 0,
                path: path.to_string(),
                link: None,
            }),
            ..Default::default()
        };
        DatasetConfig::resolve(config, &Default::default()).unwrap()
    }

    /// Parts of *different* sizes must still map global row -> (part, row)
    /// correctly; this is exactly where a `rows_per_part` assumption breaks.
    #[test]
    fn maps_global_rows_across_uneven_parts() {
        let dir = tempfile::tempdir().unwrap();
        // 4 + 7 + 2 rows: no uniform part size exists.
        for (i, rows) in [4usize, 7, 2].iter().enumerate() {
            let path = dir.path().join(format!("p_{i}.npy"));
            std::fs::write(&path, make_ramp_npy(i * 100, *rows, 2)).unwrap();
        }

        let config = resolved(dir.path(), DatasetKind::Npy, "p_{i}.npy", 3);
        let reader = PartitionedReader::open(dir.path(), &config).unwrap();
        assert_eq!(reader.num_points(), 13);

        // First row of each part carries that part's base value.
        assert_eq!(reader.vector_at(0).unwrap(), vec![0.0, 1.0]);
        assert_eq!(reader.vector_at(4).unwrap(), vec![100.0, 101.0]);
        assert_eq!(reader.vector_at(11).unwrap(), vec![200.0, 201.0]);
        // Last row overall.
        assert_eq!(reader.vector_at(12).unwrap(), vec![202.0, 203.0]);
        assert!(reader.vector_at(13).is_err(), "past the end must not wrap");
    }

    #[test]
    fn reads_payload_rows_across_parquet_parts() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("m_0.parquet"), 0, 5, 2);
        write_parquet(&dir.path().join("m_1.parquet"), 1000, 3, 2);

        let config = resolved(dir.path(), DatasetKind::Parquet, "m_{i}.parquet", 2);
        let reader = PartitionedReader::open(dir.path(), &config).unwrap();
        assert_eq!(reader.num_points(), 8);

        assert_eq!(reader.payload_object(1).unwrap().unwrap()["id"], 1);
        assert_eq!(reader.payload_object(5).unwrap().unwrap()["id"], 1000);
        assert_eq!(reader.payload_object(7).unwrap().unwrap()["id"], 1002);
    }

    /// The sidecar must survive a re-open, and a spec change must invalidate it.
    #[test]
    fn caches_and_invalidates_the_parts_index() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..2 {
            std::fs::write(
                dir.path().join(format!("p_{i}.npy")),
                make_ramp_npy(0, 3, 2),
            )
            .unwrap();
        }

        let two = resolved(dir.path(), DatasetKind::Npy, "p_{i}.npy", 2);
        assert_eq!(
            PartitionedReader::open(dir.path(), &two)
                .unwrap()
                .num_points(),
            6
        );
        let manifest = dir.path().join(".parts-index").join("test-parts.json");
        assert!(manifest.exists(), "sizing result must be cached");

        // Re-opening with the same spec reuses it.
        assert_eq!(
            PartitionedReader::open(dir.path(), &two)
                .unwrap()
                .num_points(),
            6
        );

        // A different part count is a different spec: the stale entry must not
        // be reused, or the second part would go missing.
        let one = resolved(dir.path(), DatasetKind::Npy, "p_{i}.npy", 1);
        assert_eq!(
            PartitionedReader::open(dir.path(), &one)
                .unwrap()
                .num_points(),
            3
        );
    }

    /// A part whose real size no longer matches the sidecar must be reported,
    /// not silently used — every later row would land on the wrong point.
    #[test]
    fn rejects_a_part_that_no_longer_matches_the_cached_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("p_0.npy"), make_ramp_npy(0, 6, 2)).unwrap();
        let config = resolved(dir.path(), DatasetKind::Npy, "p_{i}.npy", 1);
        PartitionedReader::open(dir.path(), &config).unwrap();

        // Same spec (so the sidecar is reused), fewer rows on disk.
        std::fs::write(dir.path().join("p_0.npy"), make_ramp_npy(0, 3, 2)).unwrap();
        let reader = PartitionedReader::open(dir.path(), &config).unwrap();
        let err = reader.vector_at(0).unwrap_err().to_string();
        assert!(err.contains("parts index"), "{err}");
    }

    /// The point of measuring over `Range`: a remote corpus is sized without
    /// being downloaded. Both formats must manage it in one small request each.
    #[test]
    fn sizes_remote_parts_without_downloading_them() {
        let npy_a = make_ramp_npy(0, 900, 16);
        let npy_b = make_ramp_npy(0, 350, 16);
        let corpus_bytes = npy_a.len() + npy_b.len();
        let (base, server) = crate::dataset::test_http::serve_ranges(
            vec![
                ("p_0.npy".to_string(), npy_a),
                ("p_1.npy".to_string(), npy_b),
            ],
            2,
        );

        let dir = tempfile::tempdir().unwrap();
        let config = DatasetConfig {
            name: "remote-parts".to_string(),
            kind: Some(DatasetKind::Npy),
            parts: Some(PartsConfig {
                count: 2,
                start: 0,
                path: "p_{i}.npy".to_string(),
                link: Some(format!("{base}/p_{{i}}.npy")),
            }),
            ..Default::default()
        };
        let config = DatasetConfig::resolve(config, &Default::default()).unwrap();

        let reader = PartitionedReader::open(dir.path(), &config).unwrap();
        assert_eq!(reader.num_points(), 1250, "900 + 350 rows");

        let stats = server.join().unwrap();
        assert_eq!(stats.requests, 2, "one ranged request per part");
        // The cost is a fixed header probe per part, not a function of how big
        // the parts are — that is what makes sizing 410 LAION parts viable.
        assert!(
            stats.bytes_served <= 2 * NPY_HEADER_PROBE && stats.bytes_served < corpus_bytes,
            "sizing served {} bytes of a {corpus_bytes}-byte corpus",
            stats.bytes_served
        );
        // Nothing was written to the datasets dir either.
        assert!(!dir.path().join("p_0.npy").exists());
    }

    /// A remote parquet part is sized from its footer, which lives at the *end*
    /// of the file — so this exercises the suffix-range path specifically.
    #[test]
    fn sizes_a_remote_parquet_part_from_its_footer() {
        let dir = tempfile::tempdir().unwrap();
        let built = dir.path().join("built.parquet");
        write_parquet(&built, 0, 3000, 256);
        let body = std::fs::read(&built).unwrap();
        std::fs::remove_file(&built).unwrap();
        let total = body.len();

        let (base, server) =
            crate::dataset::test_http::serve_ranges(vec![("m_0.parquet".to_string(), body)], 1);

        let config = DatasetConfig {
            name: "remote-parquet".to_string(),
            kind: Some(DatasetKind::Parquet),
            parts: Some(PartsConfig {
                count: 1,
                start: 0,
                path: "m_{i}.parquet".to_string(),
                link: Some(format!("{base}/m_{{i}}.parquet")),
            }),
            ..Default::default()
        };
        let config = DatasetConfig::resolve(config, &Default::default()).unwrap();

        let reader = PartitionedReader::open(dir.path(), &config).unwrap();
        assert_eq!(reader.num_points(), 3000);

        let stats = server.join().unwrap();
        assert_eq!(stats.requests, 1);
        assert!(
            stats.bytes_served <= PARQUET_TAIL_PROBE && stats.bytes_served < total,
            "the footer request served {} bytes of a {total}-byte file",
            stats.bytes_served
        );
    }

    #[test]
    fn expands_templates() {
        assert_eq!(expand("img_emb_{i}.npy", 0), "img_emb_0.npy");
        assert_eq!(expand("a/{i}/b_{i}.npy", 7), "a/7/b_7.npy");
    }
}
