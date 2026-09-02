use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{CommandFactory, Parser};
use tokio::runtime;

use args::{Args, Command};
use config::examples::ExampleKind;
use results::{BenchmarkResults, IndexPhase};

mod args;
mod client;
mod collection;
mod config;
mod dataset;
mod fbin_reader;
mod generators;
mod processor;
mod query;
mod results;
mod save_jsonl;
mod scroll;
mod search;
mod self_update;
mod serverless;
mod stats;
mod upload;
mod upsert;

/// Wait for the index and record how long it took.
async fn run_wait_index(args: &Args, stopped: Arc<AtomicBool>) -> Result<IndexPhase> {
    println!("Waiting for index to be ready...");
    let wait_secs = collection::wait_index(args, stopped).await?;
    println!("Index ready in {wait_secs} seconds");
    Ok(IndexPhase { wait_secs })
}

/// `bfb scroll --file` / `--example`: YAML-config-driven scroll.
async fn run_scroll(
    args: Args,
    scroll_args: args::ScrollArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let resolved = config::examples::resolve(
        scroll_args.config.file.as_deref(),
        scroll_args.config.example.as_deref(),
        ExampleKind::Scroll,
    )?;
    let config = config::scroll::parse(&resolved.yaml, &resolved.origin)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();

    let mut results = BenchmarkResults::new(&args, Some(resolved.origin));
    results.results.scroll = Some(query::scroll_with_config(&args, &config, stopped).await?);
    results.write_if_requested(&args)
}

/// `bfb search --file` / `--example`: YAML-config-driven search.
async fn run_search(
    args: Args,
    search_args: args::SearchArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let resolved = config::examples::resolve(
        search_args.config.file.as_deref(),
        search_args.config.example.as_deref(),
        ExampleKind::Search,
    )?;
    let config = config::search::parse(&resolved.yaml, &resolved.origin)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();

    if args.search_quality && args.search_exact {
        println!("Ignoring `exact` flag because `search_quality` is also enabled!");
    }

    let mut results = BenchmarkResults::new(&args, Some(resolved.origin));
    results.results.search = Some(query::search_with_config(&args, &config, stopped).await?);
    results.write_if_requested(&args)
}

/// `bfb upload --file` / `--example`: YAML-config-driven upload.
async fn run_upload(
    args: Args,
    upload_args: args::UploadArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let resolved = config::examples::resolve(
        upload_args.config.file.as_deref(),
        upload_args.config.example.as_deref(),
        ExampleKind::Upload,
    )?;
    let config = config::parse(&resolved.yaml, &resolved.origin)?;

    let mut args = args;
    args.num_vectors = Some(dataset::resolve_num_vectors(
        args.num_vectors,
        args.offset,
        &config,
        &dataset::default_datasets_dir(),
    )?);
    args.collection_name = config.collection.name.clone();
    if let Some(sharding) = &config.collection.sharding {
        args.shard_key = Some(sharding.key.clone());
    }

    let mut results = BenchmarkResults::new(&args, Some(resolved.origin));

    if !args.skip_create && !args.skip_setup {
        collection::recreate_collection_from_config(&config, &args, stopped.clone()).await?;
    }

    if !args.skip_upload && !args.skip_setup {
        results.results.upload =
            Some(upload::upload_with_config(&args, &config, stopped.clone()).await?);
    }

    if !args.skip_wait_index && !args.skip_setup {
        results.results.index = Some(run_wait_index(&args, stopped.clone()).await?);
    }

    results.write_if_requested(&args)
}

async fn run_benchmark(args: Args, stopped: Arc<AtomicBool>) -> Result<()> {
    match args.command.clone() {
        Some(Command::Search(search_args)) => return run_search(args, search_args, stopped).await,
        Some(Command::Upload(upload_args)) => return run_upload(args, upload_args, stopped).await,
        Some(Command::Scroll(scroll_args)) => return run_scroll(args, scroll_args, stopped).await,
        Some(Command::Serverless(serverless_args)) => {
            return serverless::run(args, serverless_args, stopped).await;
        }
        // `Schema` / `SelfUpdate` / `Completions` are handled before the runtime
        // starts; `None` falls through.
        Some(
            Command::Schema
            | Command::Examples(_)
            | Command::SelfUpdate(_)
            | Command::Completions { .. },
        )
        | None => {}
    }

    if args.search_quality && args.search_exact {
        println!("Ignoring `exact` flag because `search_quality` is also enabled!");
    }

    let mut results = BenchmarkResults::new(&args, None);

    if !args.skip_create && !args.skip_setup {
        collection::recreate_collection(&args, stopped.clone()).await?;
    }

    if !args.skip_upload && !args.skip_setup {
        results.results.upload = Some(upload::upload_data(&args, stopped.clone()).await?);
    }

    if !args.skip_wait_index && !args.skip_setup {
        results.results.index = Some(run_wait_index(&args, stopped.clone()).await?);
    }

    if args.search || args.search_quality {
        results.results.search = Some(query::search(&args, stopped.clone()).await?);
    }

    if args.scroll {
        results.results.scroll = Some(query::scroll(&args, stopped.clone()).await?);
    }

    results.write_if_requested(&args)
}

fn main() {
    let args = Args::parse();

    // Commands that need no Qdrant connection / Tokio runtime.
    match &args.command {
        Some(Command::Completions { shell }) => {
            clap_complete::generate(*shell, &mut Args::command(), "bfb", &mut std::io::stdout());
            return;
        }
        Some(Command::Schema) => {
            config::schema::print_schema();
            return;
        }
        Some(Command::Examples(examples_args)) => {
            if let Err(err) = config::examples::run(examples_args.name.as_deref()) {
                eprintln!("Error: {err:?}");
                std::process::exit(1);
            }
            return;
        }
        Some(Command::SelfUpdate(update_args)) => {
            if let Err(err) = self_update::run(update_args) {
                eprintln!("Error: {err:?}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }

    let stopped = Arc::new(AtomicBool::new(false));
    let r = stopped.clone();

    ctrlc::set_handler(move || {
        r.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let runtime = runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime");

    if let Err(err) = runtime.block_on(run_benchmark(args, stopped)) {
        eprintln!("Error: {err:?}");
        std::process::exit(1);
    }
}
