use ahash::HashMap;
use clap::Args;
use hnswlib_rs::HnswIndex;
use itertools::Itertools;
use libroxanne::blob::BlobStore;
use libroxanne::cfg::RoxanneDbCfg;
use libroxanne::common::StdMetric;
use libroxanne::db::Db;
use libroxanne::db::DbIndexMode;
use libroxanne::db::DbTransaction;
use libroxanne::db::NodeData;
use libroxanne::RoxanneDbDir;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Args)]
pub struct MigrateHnswArgs {
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
}

impl MigrateHnswArgs {
  pub async fn exec(self: MigrateHnswArgs) {
    let dir = RoxanneDbDir::new(self.out);

    // Make sure database can be created before we do long expensive work.
    let db = Db::open(dir.db()).await;
    tracing::info!("created database");

    let raw = std::fs::File::open(&self.path).unwrap();
    let mut rd = BufReader::new(raw);
    let index = HnswIndex::load(self.dim, &mut rd);

    let cfg = RoxanneDbCfg {
      degree_bound: index.m,
      dim: self.dim,
      metric: self.metric,
      query_search_list_cap: index.ef,
      update_search_list_cap: index.ef_construction,
      ..Default::default()
    };
    tokio::fs::write(dir.cfg(), toml::to_string(&cfg).unwrap())
      .await
      .unwrap();

    // In HNSW, "labels" are the external ID that the builder has defined for each vector.
    // To port to Roxanne, they will become the keys. We'll assign each a new internal Roxanne ID.
    let hnsw_label_to_rx_id = index
      .labels()
      .enumerate()
      .map(|(id, l)| (l, id))
      .collect::<HashMap<_, _>>();
    let mut vectors = index
      .labels()
      .map(|hnsw_label| {
        (
          hnsw_label_to_rx_id[&hnsw_label],
          index.get_data_by_label(hnsw_label).to_vec(),
        )
      })
      .collect::<HashMap<_, _>>();
    let mut graph = index
      .labels()
      .map(|label| {
        let neighbors = index
          .get_merged_neighbors(label, 0)
          .into_iter()
          .map(|hnsw_label| hnsw_label_to_rx_id[&hnsw_label])
          .collect_vec();
        let rx_id = hnsw_label_to_rx_id[&label];
        (rx_id, neighbors)
      })
      .collect::<HashMap<_, _>>();

    // TODO Copy deleted markers.
    let mut txn = DbTransaction::new();
    txn.write_index_mode(DbIndexMode::InMemory);
    txn.write_medoid(hnsw_label_to_rx_id[&index.entry_label()]);
    for (label, id) in hnsw_label_to_rx_id {
      let key = label.to_string();
      txn.write_id(&key, id);
      txn.write_key(id, &key);
      txn.write_node(id, &NodeData {
        neighbors: graph.remove(&id).unwrap(),
        vector: vectors.remove(&id).unwrap(),
      });
    }
    txn.commit(&db);
    db.flush().await;
    drop(db);
    tracing::info!("all done!");
  }
}
