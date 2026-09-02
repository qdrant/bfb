//! `bfb serverless clear` — delete collections matching a prefix.

use anyhow::Result;

use super::args::ServerlessClearArgs;
use super::client::single_client;
use super::collections::list_matching;
use crate::args::Args;

pub async fn run(args: &Args, clear: ServerlessClearArgs) -> Result<()> {
    let client = single_client(args)?;
    let names: Vec<String> = list_matching(&client, &clear.collection_prefix)
        .await?
        .into_iter()
        .map(|c| c.collection_name)
        .collect();

    if names.is_empty() {
        println!(
            "No collections matched prefix {:?}",
            clear.collection_prefix
        );
        return Ok(());
    }

    println!(
        "Deleting {} collection(s) with prefix {:?}",
        names.len(),
        clear.collection_prefix
    );

    let mut deleted = 0usize;
    for name in &names {
        let ok = client.delete_collection(name).await?;
        if ok {
            deleted += 1;
            println!("  deleted {name}");
        } else {
            println!("  skipped {name} (already gone)");
        }
    }
    println!("Deleted {deleted}/{} collections", names.len());
    Ok(())
}
