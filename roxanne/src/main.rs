use clap::Parser;
use clap::Subcommand;
use roxanne::cmd::eval::EvalArgs;
use roxanne::cmd::export_vectors::ExportVectorsArgs;
use roxanne::cmd::migrate_hnsw::MigrateHnswArgs;
use roxanne::cmd::migrate_sharded_hnsw::MigrateShardedHnswArgs;
use roxanne::cmd::serve::ServeArgs;

#[derive(Subcommand)]
enum Commands {
  /// Evaluate queries against a Roxanne database.
  Eval(EvalArgs),
  /// Export all vectors from a Roxanne database.
  ExportVectors(ExportVectorsArgs),
  /// Create a new Roxanne database from an existing HNSW index.
  MigrateHnsw(MigrateHnswArgs),
  /// Create a new Roxanne database from a set of existing HNSW uniform-shard indices.
  MigrateShardedHnsw(MigrateShardedHnswArgs),
  /// Serve a Roxanne database over HTTP.
  Serve(ServeArgs),
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
    Commands::MigrateShardedHnsw(args) => args.exec().await,
    Commands::Serve(args) => args.exec().await,
  };
}
