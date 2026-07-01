mod config;
mod download;
mod payload;
mod reader;
mod readers;
mod registry;
mod sources;
mod upload;

pub use config::DatasetConfig;
pub use sources::UploadDatasetSources;
pub use upload::resolve_num_vectors;

/// Default directory for downloaded datasets (matches vector-db-benchmark layout).
pub fn default_datasets_dir() -> std::path::PathBuf {
    std::env::var_os("BFB_DATASETS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("datasets"))
}
