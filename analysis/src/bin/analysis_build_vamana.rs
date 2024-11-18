use clap::Parser;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use libroxanne::common::PrecomputedDists;
use libroxanne::common::StdMetric;
use libroxanne::in_memory::InMemoryIndex;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use std::fs;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
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

  #[arg(long, default_value_t = StdMetric::L2)]
  metric: StdMetric,

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

  let out_dir = format!(
    "vamana-{}M-{}ef-{}a",
    args.degree_bound, args.search_list_cap, args.distance_threshold
  );
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  let vecs = ds.read_vectors();
  println!("Loaded vectors");
  let n = ds.info.n;
  let precomputed_dists = if args.load_precomputed_dists {
    let dists = ds.read_dists();
    println!("Loaded dists");
    Some(Arc::new(PrecomputedDists::new(
      (0..n).map(|i| (i, i)).collect(),
      dists,
    )))
  } else {
    None
  };

  let pb = new_pb(n * 2); // There are two passes.
  let index = InMemoryIndex::builder(
    (0..n).collect(),
    (0..n).map(|i| vecs.row(i).to_owned()).collect(),
  )
  .metric(args.metric.get_fn())
  .degree_bound(args.degree_bound)
  .distance_threshold(args.distance_threshold)
  .update_batch_size(args.update_batch_size)
  .update_search_list_cap(args.search_list_cap)
  .medoid_sample_size(args.medoid_sample_size)
  .precomputed_dists(precomputed_dists)
  .on_progress(|completed| pb.set_position(completed as u64))
  .build();
  pb.finish();
  println!("Built graph");

  export_index(&ds, &out_dir, &index.graph, index.medoid);
}
