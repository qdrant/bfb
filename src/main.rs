use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{CommandFactory, FromArgMatches};
use tokio::runtime;

use args::Args;

mod args;
mod client;
mod collection;
mod common;
mod fbin_reader;
mod processor;
mod query;
mod save_jsonl;
mod scroll;
mod search;
mod stats;
mod upload;
mod upsert;

async fn run_benchmark(args: Args, stopped: Arc<AtomicBool>) -> Result<()> {
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

    let stopped = Arc::new(AtomicBool::new(false));
    let r = stopped.clone();

    ctrlc::set_handler(move || {
        r.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let runtime = runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build();

    runtime
        .unwrap()
        .block_on(run_benchmark(args, stopped))
        .unwrap();
}
