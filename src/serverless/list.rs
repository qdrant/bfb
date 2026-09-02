//! `bfb serverless list` — collections matching a prefix and their point counts.

use anyhow::Result;

use super::args::ServerlessListArgs;
use super::client::single_client;
use super::collections::list_matching;
use crate::args::Args;

pub async fn run(args: &Args, list: ServerlessListArgs) -> Result<()> {
    let client = single_client(args)?;
    let summaries = list_matching(&client, &list.collection_prefix).await?;

    if summaries.is_empty() {
        if list.collection_prefix.is_empty() {
            println!("No collections in the space");
        } else {
            println!("No collections matched prefix {:?}", list.collection_prefix);
        }
        return Ok(());
    }

    let width = summaries
        .iter()
        .map(|c| c.collection_name.len())
        .max()
        .unwrap_or(0);
    let mut total: u64 = 0;
    let mut unknown = 0usize;
    for c in &summaries {
        match c.point_count {
            Some(n) => {
                total += n;
                println!("{:<width$}  {n}", c.collection_name);
            }
            None => {
                unknown += 1;
                println!("{:<width$}  ?", c.collection_name);
            }
        }
    }

    print!("{} collection(s), {total} points", summaries.len());
    if unknown > 0 {
        print!(" (+{unknown} with unknown count)");
    }
    println!();
    Ok(())
}
