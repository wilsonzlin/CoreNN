use clap::Parser;
use itertools::Itertools;
use libroxanne::common::metric_euclidean;
use libroxanne::in_memory::calc_approx_medoid;
use libroxanne::in_memory::random_r_regular_graph;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long)]
  degree_bound: usize,
}

fn main() {
  let ds = Dataset::init();
  let n = ds.info.n;

  let args = Args::parse();

  let out_dir = format!("random-{}", args.degree_bound);
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  let vecs = ds.read_vectors();

  let graph = random_r_regular_graph(&(0..ds.info.n).collect_vec(), args.degree_bound);

  let medoid = calc_approx_medoid(
    &(0..n).map(|id| (id, vecs.row(id).to_owned())).collect(),
    metric_euclidean,
    10_000,
    None,
  );

  export_index(&ds, &out_dir, &graph, medoid);
}
