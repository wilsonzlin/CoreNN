use clap::Parser;
use hnswlib_rs::HnswIndex;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use libroxanne::common::StdMetric;
use libroxanne::db::Db;
use libroxanne::db::NodeData;
use libroxanne::vamana::VamanaParams;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to a HNSW index file.
  #[arg()]
  path: PathBuf,

  /// Output directory to write Roxanne index to.
  #[arg(long)]
  out: PathBuf,

  /// Dimensions of the vectors.
  #[arg(long)]
  dim: usize,

  /// Metric.
  #[arg(long)]
  metric: StdMetric,

  #[arg(long, default_value_t = 4)]
  beam_width: usize,

  #[arg(long, default_value_t = 80)]
  degree_bound: usize,

  #[arg(long, default_value_t = 1.1)]
  distance_threshold: f64,

  #[arg(long, default_value_t = 150)]
  query_search_list_cap: usize,

  #[arg(long, default_value_t = 64)]
  update_batch_size: usize,

  #[arg(long, default_value_t = 300)]
  update_search_list_cap: usize,
}

fn load_hnsw(dim: usize, path: impl AsRef<Path>) -> HnswIndex {
  let raw = File::open(path).unwrap();
  let mut rd = BufReader::new(raw);
  HnswIndex::load(dim, &mut rd)
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

// We don't perform PQ here, as it's expensive and not specifically related to this task; use the general `pqify` tool afterwards.
fn main() {
  let args = Args::parse();

  // Make sure database can be created before we do long expensive work.
  let db = Db::create(&args.out);
  println!("Created database");

  let index = load_hnsw(args.dim, args.path);

  // Allow custom params for new graph that will be stored on disk, which means the params might be different from the HNSW index (so don't just copy existing).
  // TODO beam_width, query_search_list_cap.
  let cfg = VamanaParams {
    degree_bound: args.degree_bound,
    distance_threshold: args.distance_threshold,
    update_batch_size: args.update_batch_size,
    update_search_list_cap: args.update_search_list_cap,
  };

  db.write_cfg(&cfg);
  db.write_dim(args.dim);
  db.write_medoid(index.entry_label());
  db.write_metric(args.metric);

  let pb = new_pb(index.cur_element_count);
  // Collect to Vec so we can use into_par_iter, which is much faster than par_bridge.
  index.labels().collect_vec().into_par_iter().for_each(|id| {
    let node_data = NodeData {
      neighbors: index.get_merged_neighbors(id, 0).into_iter().collect_vec(),
      vector: index.get_data_by_label(id),
    };
    db.write_node(id, &node_data);
    pb.inc(1);
  });
  pb.finish();
  println!("Finalizing database");

  db.flush();
  drop(db);
  println!("All done!");
}
