use clap::Parser;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::metric_euclidean;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use std::fs;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long)]
  degree_bound: usize,

  #[arg(long, default_value_t = 1.1)]
  distance_threshold: f64,

  #[arg(long, default_value_t = 64)]
  update_batch_size: usize,

  #[arg(long)]
  search_list_cap: usize,

  #[arg(long, default_value_t = 10_000)]
  medoid_sample_size: usize,

  #[arg(long)]
  load_precomputed_dists: bool,
}

fn new_pb(len: usize) -> ProgressBar {
  let pb = ProgressBar::new(len.try_into().unwrap());
  pb.set_style(
    ProgressStyle::with_template(
      "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )
    .unwrap()
    .progress_chars("#>-"),
  );
  pb
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: args.distance_threshold,
    query_search_list_cap: 1, // Irrelevant.
    update_batch_size: args.update_batch_size,
    update_search_list_cap: args.search_list_cap,
  };
  println!("Params: {params:?}");

  let out_dir = format!(
    "vamana-{}M-{}ef-{}a",
    params.degree_bound, params.update_search_list_cap, params.distance_threshold
  );
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  let vecs = ds.read_vectors();
  println!("Loaded vectors");
  let n = ds.info.n;
  let precomputed_dists = if args.load_precomputed_dists {
    let dists = ds.read_dists();
    println!("Loaded dists");
    Some(dists)
  } else {
    None
  };

  let pb = new_pb(n);
  let index = InMemoryVamana::build_index(
    (0..n).map(|i| (i, vecs.row(i).to_owned())).collect(),
    metric_euclidean,
    params,
    args.medoid_sample_size,
    precomputed_dists.map(|pd| Arc::new(((0..n).map(|i| (i, i)).collect(), pd))),
    |completed, _metrics| pb.set_position(completed as u64),
  );
  pb.finish();
  println!("Built graph");

  export_index(&ds, &out_dir, index.datastore().graph(), index.medoid());
}
