use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{CommandFactory, FromArgMatches};
use tokio::runtime;

use args::{Args, Command};
use results::{BenchmarkResults, IndexPhase};

mod args;
mod client;
mod collection;
mod config;
mod dataset;
mod fbin_reader;
mod generators;
mod memory;
mod processor;
mod query;
mod results;
mod save_jsonl;
mod scroll;
mod search;
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

/// The REST base URL used for endpoints gRPC does not expose. Derived from
/// `--uri` by mapping Qdrant's gRPC port to its REST one.
fn rest_uri(args: &Args) -> String {
    memory::rest_url_from_grpc(args.uri.first().expect("clap guarantees one --uri"))
}

/// Sample memory and disk usage. Best-effort: a REST failure warns rather than
/// failing a run that already has its headline numbers.
async fn fetch_memory(args: &Args) -> Option<memory::MemoryReport> {
    if args.skip_server_stats {
        return None;
    }
    let (rest_uri, collection) = (rest_uri(args), args.collection_name.clone());
    let api_key = std::env::var("QDRANT_API_KEY").ok();

    // `ureq` is blocking; keep it off the async worker threads.
    let fetched = tokio::task::spawn_blocking(move || {
        memory::fetch(&rest_uri, &collection, api_key.as_deref())
    })
    .await;

    match fetched {
        Ok(Ok(report)) => {
            report.print();
            Some(report)
        }
        Ok(Err(err)) => {
            eprintln!("Warning: could not read memory usage: {err:#}");
            eprintln!("         (pass --skip-server-stats to silence)");
            None
        }
        Err(err) => {
            eprintln!("Warning: memory task failed: {err}");
            None
        }
    }
}

/// `bfb scroll --file config.yaml`: YAML-config-driven scroll.
async fn run_scroll(
    args: Args,
    scroll_args: args::ScrollArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let config = config::scroll::load(&scroll_args.file)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();

    let mut results = BenchmarkResults::new(&args, Some(scroll_args.file.clone()));
    results.results.scroll = Some(query::scroll_with_config(&args, &config, stopped).await?);
    results.write_if_requested(&args)
}

/// `bfb search --file config.yaml`: YAML-config-driven search.
async fn run_search(
    args: Args,
    search_args: args::SearchArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let config = config::search::load(&search_args.file)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();

    if args.search_quality && args.search_exact {
        println!("Ignoring `exact` flag because `search_quality` is also enabled!");
    }

    let mut results = BenchmarkResults::new(&args, Some(search_args.file.clone()));
    results.results.search = Some(query::search_with_config(&args, &config, stopped).await?);
    results.write_if_requested(&args)
}

/// `bfb upload --file config.yaml`: YAML-config-driven upload.
async fn run_upload(
    args: Args,
    upload_args: args::UploadArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let config = config::load(&upload_args.file)?;

    let mut args = args;
    args.num_vectors = Some(dataset::resolve_num_vectors(
        args.num_vectors,
        &config,
        &dataset::default_datasets_dir(),
    )?);
    args.collection_name = config.collection.name.clone();
    if let Some(sharding) = &config.collection.sharding {
        args.shard_key = Some(sharding.key.clone());
    }

    let mut results = BenchmarkResults::new(&args, Some(upload_args.file.clone()));

    if !args.skip_create && !args.skip_setup {
        collection::recreate_collection_from_config(&config, &args, stopped.clone()).await?;
    }

    if !args.skip_upload && !args.skip_setup {
        results.results.upload =
            Some(upload::upload_with_config(&args, &config, stopped.clone()).await?);
    }

    if !args.skip_wait_index && !args.skip_setup {
        results.results.index = Some(run_wait_index(&args, stopped.clone()).await?);
        results.results.memory = fetch_memory(&args).await;
    }

    results.write_if_requested(&args)
}

async fn run_benchmark(args: Args, stopped: Arc<AtomicBool>) -> Result<()> {
    match args.command.clone() {
        Some(Command::Search(search_args)) => return run_search(args, search_args, stopped).await,
        Some(Command::Upload(upload_args)) => return run_upload(args, upload_args, stopped).await,
        Some(Command::Scroll(scroll_args)) => return run_scroll(args, scroll_args, stopped).await,
        // `Schema` is handled before the runtime starts; `None` falls through.
        Some(Command::Schema) | None => {}
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
        results.results.memory = fetch_memory(&args).await;
    }

    if args.search || args.search_quality {
        results.results.search = Some(query::search(&args, stopped.clone()).await?);
    }

    if args.scroll {
        results.results.scroll = Some(query::scroll(&args, stopped.clone()).await?);
    }

    results.write_if_requested(&args)
}

/// Parse command line arguments with shell completion installation support
fn parse_args() -> Args {
    // Create the command and add shell completion subcommand
    let mut command = Args::command();
    command = clap_autocomplete::add_subcommand(command);

    // Hide the complete subcommand from help to keep main command clean
    if let Some(complete_cmd) = command.find_subcommand_mut("complete") {
        *complete_cmd = complete_cmd.clone().hide(true);
    }

    // Parse arguments
    let matches = command.clone().get_matches();

    // Check if the complete subcommand was used
    if let Some(result) = clap_autocomplete::test_subcommand(&matches, command) {
        if let Err(err) = result {
            eprintln!("Insufficient permissions: {err}");
            std::process::exit(1);
        } else {
            std::process::exit(0);
        }
    }

    // Parse args normally for the main application logic
    Args::from_arg_matches(&matches).unwrap()
}

fn main() {
    let args = parse_args();

    // Pure print commands that need no network / Tokio runtime.
    if let Some(Command::Schema) = args.command {
        config::schema::print_schema();
        return;
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
