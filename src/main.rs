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

/// The bound applied to the best-effort memory REST call: the CLI `--timeout` if
/// set, otherwise `memory`'s own fallback.
fn memory_http_timeout(args: &Args) -> Option<std::time::Duration> {
    args.timeout
        .map(|secs| std::time::Duration::from_secs(secs as u64))
}

/// Sample memory and disk usage. Best-effort: a REST failure warns rather than
/// failing a run that already has its headline numbers.
async fn fetch_memory(args: &Args) -> Option<memory::MemoryReport> {
    if args.skip_server_stats {
        return None;
    }
    // Map every `--uri` to its REST port and try each in turn: the client spreads
    // work across all URIs, so the first may be down while another is healthy.
    let rest_uris: Vec<String> = args
        .uri
        .iter()
        .map(|u| memory::rest_url_from_grpc(u))
        .collect();
    let collection = args.collection_name.clone();
    let api_key = std::env::var("QDRANT_API_KEY").ok();
    let timeout = memory_http_timeout(args);

    // `ureq` is blocking; keep it off the async worker threads.
    let fetched = tokio::task::spawn_blocking(move || {
        let mut last_err = None;
        for rest_uri in &rest_uris {
            match memory::fetch(rest_uri, &collection, api_key.as_deref(), timeout) {
                Ok(report) => return Ok(report),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.expect("clap guarantees at least one --uri"))
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

    let mut results = BenchmarkResults::new(
        &args,
        Some(scroll_args.file.clone()),
        Some(args.num_vectors_or_default()),
    );
    results.results.scroll = Some(query::scroll_with_config(&args, &config, stopped).await?);
    results.write_if_requested(&args)
}

/// `bfb create-field-index`: build field indices on a populated collection.
async fn run_create_field_index(
    args: Args,
    create_args: args::CreateFieldIndexArgs,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let config = config::load(&create_args.file)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();
    let mut specs = collection::field_index_specs_from_config(&config);

    if !create_args.field.is_empty() {
        let wanted = &create_args.field;
        for name in wanted {
            if !specs.iter().any(|spec| &spec.field == name) {
                anyhow::bail!("--field {name:?} is not a declared, indexed field");
            }
        }
        specs.retain(|spec| wanted.contains(&spec.field));
    }

    if specs.is_empty() {
        anyhow::bail!("no field indices to create: the config declares none with `index: true`");
    }

    let client = client::random_client(&args)?;
    let phase = collection::create_field_indices(&client, specs).await?;

    println!("--- Field index creation ---");
    for field in &phase.fields {
        println!(
            "{} ({}): {:.3} s (server {:.3} s)",
            field.field, field.kind, field.duration_secs, field.server_secs
        );
    }
    println!("Total: {:.3} s", phase.duration_secs);

    // Phase-only: neither uploads nor queries, so no requested point count.
    let mut results = BenchmarkResults::new(&args, Some(create_args.file.clone()), None);
    results.results.create_field_index = Some(phase);

    // `create_field_index(wait=true)` returns once the field index exists, but any
    // optimizer work it schedules — notably the extra HNSW links for that field —
    // runs afterwards. Waiting for it is what folds that cost into the index wait.
    if !args.skip_wait_index {
        results.results.index = Some(run_wait_index(&args, stopped).await?);
        results.results.memory = fetch_memory(&args).await;
    }

    results.write_if_requested(&args)
}

/// `bfb drop-field-index`: remove field indices so they can be rebuilt and re-measured.
async fn run_drop_field_index(args: Args, drop_args: args::DropFieldIndexArgs) -> Result<()> {
    let config = config::load(&drop_args.file)?;

    let mut args = args;
    args.collection_name = config.collection.name.clone();
    let fields: Vec<String> = if drop_args.field.is_empty() {
        collection::field_index_specs_from_config(&config)
            .into_iter()
            .map(|spec| spec.field)
            .collect()
    } else {
        drop_args.field.clone()
    };

    if fields.is_empty() {
        anyhow::bail!("no field indices to drop");
    }

    let client = client::random_client(&args)?;
    collection::drop_field_indices(&client, &args.collection_name, &fields).await
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

    let mut results = BenchmarkResults::new(
        &args,
        Some(search_args.file.clone()),
        Some(args.num_vectors_or_default()),
    );
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

    let mut results = BenchmarkResults::new(
        &args,
        Some(upload_args.file.clone()),
        Some(args.num_vectors_or_default()),
    );

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
        Some(Command::CreateFieldIndex(create_args)) => {
            return run_create_field_index(args, create_args, stopped).await;
        }
        Some(Command::DropFieldIndex(drop_args)) => {
            return run_drop_field_index(args, drop_args).await;
        }
        // `Schema` is handled before the runtime starts; `None` falls through.
        Some(Command::Schema) | None => {}
    }

    if args.search_quality && args.search_exact {
        println!("Ignoring `exact` flag because `search_quality` is also enabled!");
    }

    let mut results = BenchmarkResults::new(&args, None, Some(args.num_vectors_or_default()));

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
