use clap::Parser;
use clap::Subcommand;
use roxanne::cmd::eval::EvalArgs;
use roxanne::cmd::export_vectors::ExportVectorsArgs;
use roxanne::cmd::migrate_hnsw::MigrateHnswArgs;

#[derive(Subcommand)]
enum Commands {
  /// Evaluate queries against a Roxanne database.
  Eval(EvalArgs),
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
    Commands::Eval(args) => args.exec().await,
    Commands::ExportVectors(args) => args.exec().await,
    Commands::MigrateHnsw(args) => args.exec().await,
  };
}
