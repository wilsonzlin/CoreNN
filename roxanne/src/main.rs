use clap::Parser;
use clap::Subcommand;
use roxanne::cmd::eval::EvalArgs;
use roxanne::cmd::export_vectors::ExportVectorsArgs;
use roxanne::cmd::migrate_hnsw::MigrateHnswArgs;
use roxanne::cmd::serve::ServeArgs;
use tikv_jemallocator::Jemalloc;

extern crate blas_src;

// Use jemalloc as the GNU malloc doesn't return freed memory to system during full-database iteration. Also see https://github.com/facebook/rocksdb/issues/12425#issuecomment-2004733577.
#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

#[derive(Subcommand)]
enum Commands {
  /// Evaluate queries against a Roxanne database.
  Eval(EvalArgs),
  /// Export all vectors from a Roxanne database.
  ExportVectors(ExportVectorsArgs),
  /// Create a new Roxanne database from an existing HNSW index.
  MigrateHnsw(MigrateHnswArgs),
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
    Commands::Serve(args) => args.exec().await,
  };
}
