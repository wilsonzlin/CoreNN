#![feature(async_closure)]

pub mod cmd;

use clap::Parser;
use clap::Subcommand;
use cmd::export_vectors::ExportVectorsArgs;
use cmd::migrate_hnsw::MigrateHnswArgs;
use cmd::migrate_hnsw::{self};
use std::path::PathBuf;

#[derive(Subcommand)]
enum Commands {
  /// Export all vectors from a Roxanne database.
  ExportVectors(ExportVectorsArgs),
  /// Create a new Roxanne database from an existing HNSW index.
  MigrateHnsw(MigrateHnswArgs),
}

#[derive(Parser)]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[tokio::main]
async fn main() {
  tracing_subscriber::fmt::init();

  let cli = Cli::parse();
  match cli.command {
    Commands::ExportVectors(args) => args.exec().await,
    Commands::MigrateHnsw(args) => args.exec().await,
  };
}
