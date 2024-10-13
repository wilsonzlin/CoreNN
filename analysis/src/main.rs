use clap::Parser;
use std::path::PathBuf;

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Path to the array of u32 IDs.
  #[arg(long)]
  ids: PathBuf,

  /// Path to the matrix of f32 vectors.
  #[arg(long)]
  vecs: PathBuf,
}

fn main() {
  let args = Args::parse();
}
