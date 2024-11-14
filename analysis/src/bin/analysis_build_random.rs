use clap::Parser;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
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

  let args = Args::parse();

  let out_dir = format!("random-{}", args.degree_bound);
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  let vecs = ds.read_vectors();

  let params = VamanaParams {
    beam_width: 1, // Irrelevant.
    degree_bound: args.degree_bound,
    distance_threshold: 1.0,   // Irrelevant.
    query_search_list_cap: 1,  // Irrelevant.
    update_batch_size: 1,      // Irrelevant.
    update_search_list_cap: 1, // Irrelevant.
  };

  let index = InMemoryVamana::init_random_index(
    (0..ds.info.n)
      .map(|i| (i, vecs.row(i).to_owned()))
      .collect(),
    metric_euclidean,
    params,
    10_000,
    None,
  );

  export_index(&ds, &out_dir, index.datastore().graph(), index.medoid());
}
