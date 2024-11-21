use ahash::HashMap;
use clap::Parser;
use hnswlib_rs::HnswIndex;
use itertools::Itertools;
use libroxanne::blob::BlobStore;
use libroxanne::blob::InMemoryIndexBlob;
use libroxanne::cfg::RoxanneDbCfg;
use libroxanne::common::StdMetric;
use libroxanne::db::Db;
use libroxanne::db::DbTransaction;
use ndarray::Array1;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

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

fn main() {
  let args = Args::parse();

  // Make sure database can be created before we do long expensive work.
  let db = Db::open(args.out.join("db"));
  println!("Created database");

  let blobs = BlobStore::open(args.out.join("blobs"));

  let index = load_hnsw(args.dim, args.path);

  // Allow custom params for new graph that will be stored on disk, which means the params might be different from the HNSW index (so don't just copy existing).
  let cfg = RoxanneDbCfg {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    dim: args.dim,
    distance_threshold: args.distance_threshold,
    metric: args.metric,
    query_search_list_cap: args.query_search_list_cap,
    update_batch_size: args.update_batch_size,
    update_search_list_cap: args.update_search_list_cap,
    ..Default::default()
  };
  fs::write(
    args.out.join("roxanne.toml"),
    toml::to_string(&cfg).unwrap(),
  )
  .unwrap();

  // In HNSW, "labels" are the external ID that the builder has defined for each vector.
  // To port to Roxanne, they will become the keys. We'll assign each a new internal Roxanne ID.
  let label_to_id = index
    .labels()
    .enumerate()
    .map(|(id, l)| (l, id))
    .collect::<HashMap<_, _>>();
  let vectors = Arc::new(
    index
      .labels()
      .map(|label| {
        (
          label_to_id[&label],
          Array1::from_vec(index.get_data_by_label(label)),
        )
      })
      .collect(),
  );
  let graph = Arc::new(
    index
      .labels()
      .map(|label| {
        let neighbors = index
          .get_merged_neighbors(label, 0)
          .into_iter()
          .map(|label| label_to_id[&label])
          .collect_vec();
        let id = label_to_id[&label];
        (id, neighbors)
      })
      .collect(),
  );

  let blob = InMemoryIndexBlob {
    graph,
    medoid: label_to_id[&index.entry_label()],
    vectors,
  };
  blobs.write_temp_index(0, &blob);
  println!("Saved temp index");

  let mut txn = DbTransaction::new();
  txn.write_temp_index_count(1);
  for (label, id) in label_to_id {
    let key = label.to_string();
    txn.write_id(&key, id);
    txn.write_key(id, &key);
  }
  txn.commit(&db);
  db.flush();
  drop(db);
  println!("All done!");
}
