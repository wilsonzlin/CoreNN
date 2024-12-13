use clap::Parser;
use roxanne_analysis::analyze::analyze_graph;
use roxanne_analysis::Dataset;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg()]
  variant: String,

  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  /// This must be at least 100 (as k=100). For consistent benchmarking results, this should be 100.
  #[arg(long, default_value_t = 100)]
  search_list_cap: usize,

  #[arg(long)]
  subspaces: Option<usize>,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  analyze_graph(
    &ds,
    &args.variant,
    args.beam_width,
    args.search_list_cap,
    args.subspaces,
  )
}
