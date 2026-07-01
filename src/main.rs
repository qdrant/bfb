use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{CommandFactory, FromArgMatches};
use tokio::runtime;

use args::{Args, Command};

mod args;
mod client;
mod collection;
mod common;
mod config;
mod dataset;
mod fbin_reader;
mod generator;
mod processor;
mod query;
mod save_jsonl;
mod schema;
mod scroll;
mod search;
mod search_config;
mod search_generator;
mod stats;
mod upload;
mod upsert;

/// `bfb search --file config.yaml`: YAML-config-driven search.
async fn run_search(
    args: Args,
    search_args: args::SearchArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let config = search_config::load(&search_args.file)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();

    if args.search_quality && args.search_exact {
        println!("Ignoring `exact` flag because `search_quality` is also enabled!");
    }

    query::search_with_config(&args, &config, stopped).await
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

    if !args.skip_create && !args.skip_setup {
        collection::recreate_collection_from_config(&config, &args, stopped.clone()).await?;
    }

    if !args.skip_upload && !args.skip_setup {
        upload::upload_with_config(&args, &config, stopped.clone()).await?;
    }

    if !args.skip_wait_index && !args.skip_setup {
        println!("Waiting for index to be ready...");
        let wait_time = collection::wait_index(&args, stopped.clone()).await?;
        println!("Index ready in {wait_time} seconds");
    }

    Ok(())
}

async fn run_benchmark(args: Args, stopped: Arc<AtomicBool>) -> Result<()> {
    if let Some(Command::Search(search_args)) = args.command.clone() {
        return run_search(args, search_args, stopped).await;
    }

    if let Some(Command::Upload(upload_args)) = args.command.clone() {
        return run_upload(args, upload_args, stopped).await;
    }

    if args.search_quality && args.search_exact {
        println!("Ignoring `exact` flag because `search_quality` is also enabled!");
    }

    if !args.skip_create && !args.skip_setup {
        collection::recreate_collection(&args, stopped.clone()).await?;
    }

    if !args.skip_upload && !args.skip_setup {
        upload::upload_data(&args, stopped.clone()).await?;
    }

    if !args.skip_wait_index && !args.skip_setup {
        println!("Waiting for index to be ready...");
        let wait_time = collection::wait_index(&args, stopped.clone()).await?;
        println!("Index ready in {wait_time} seconds");
    }

    if args.search || args.search_quality {
        query::search(&args, stopped.clone()).await?;
    }

    if args.scroll {
        query::scroll(&args, stopped.clone()).await?;
    }

    Ok(())
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
        schema::print_schema();
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
