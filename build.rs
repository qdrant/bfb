use clap::CommandFactory;
use clap_complete::aot::Fish;
use clap_complete::generate_to;
use std::io::Error;
use std::path::PathBuf;

include!("src/args.rs");

fn main() -> Result<(), Error> {
    let home = std::env::var("HOME").expect("failed to find home directory");

    let mut cmd = Args::command();

    // Generate for fish
    // Check if fish is in PATH
    if let Ok(_) = std::process::Command::new("fish").arg("--version").output() {
        let outdir = PathBuf::from(home.clone() + "/.config/fish/completions").canonicalize()?;
        // ensure path exists
        std::fs::create_dir_all(&outdir)?;
        generate_to(Fish, &mut cmd, "bfb", outdir)?;
    }

    Ok(())
}
