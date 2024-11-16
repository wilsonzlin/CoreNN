use ahash::HashMap;
use clap::Parser;
use hnswlib_rs::HnswIndex;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use libroxanne::common::StdMetric;
use libroxanne::db::Db;
use libroxanne::db::DbTransaction;
use libroxanne::db::NodeData;
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
  let mut txn = DbTransaction::new();
  txn.write_cfg_beam_width(args.beam_width);
  txn.write_cfg_degree_bound(args.degree_bound);
  txn.write_cfg_distance_threshold(args.distance_threshold);
  txn.write_cfg_query_search_list_cap(args.query_search_list_cap);
  txn.write_cfg_update_batch_size(args.update_batch_size);
  txn.write_cfg_update_search_list_cap(args.update_search_list_cap);
  txn.write_dim(args.dim);
  txn.write_medoid(index.entry_label());
  txn.write_metric(args.metric);
  txn.write_next_id(index.cur_element_count);
  txn.write_node_count(index.cur_element_count);
  txn.write_temp_index_offsets(&[]);
  txn.commit(&db);

  let pb = new_pb(index.cur_element_count);
  let label_to_id = index
    .labels()
    .enumerate()
    .map(|(id, l)| (l, id))
    .collect::<HashMap<_, _>>();
  // Collect to Vec so we can use into_par_iter, which is much faster than par_bridge.
  index
    .labels()
    .collect_vec()
    .into_par_iter()
    .for_each(|label| {
      let node_data = NodeData {
        neighbors: index
          .get_merged_neighbors(label, 0)
          .into_iter()
          .map(|label| label_to_id[&label])
          .collect_vec(),
        vector: index.get_data_by_label(label),
      };
      let id = label_to_id[&label];
      let key = format!("{}", id);
      let mut txn = DbTransaction::new();
      txn.write_id(&key, id);
      txn.write_key(id, &key);
      txn.write_node(id, &node_data);
      txn.commit(&db);
      pb.inc(1);
    });
  pb.finish();
  println!("Finalizing database");

  db.flush();
  drop(db);
  println!("All done!");
}
