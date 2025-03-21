use crate::load_hnsw;
use ahash::HashMap;
use clap::Args;
use half::f16;
use itertools::Itertools;
use libroxanne::cfg::CfgRaw;
use libroxanne::common::StdMetric;
use libroxanne::db::Db;
use libroxanne::db::DbTransaction;
use libroxanne::db::NodeData;
use ndarray::Array1;
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
    // Make sure database can be created before we do long expensive work.
    let db = Db::open(self.out).await;
    tracing::info!("created database");

    let index = load_hnsw(self.dim, &self.path);

    let cfg = CfgRaw {
      max_edges: Some(index.m),
      metric: Some(self.metric),
      query_search_list_cap: Some(index.ef),
      update_search_list_cap: Some(index.ef_construction),
      ..Default::default()
    };

    // In HNSW, "labels" are the external ID that the builder has defined for each vector.
    // To port to Roxanne, they will become the keys. We'll assign each a new internal Roxanne ID.
    // We offset by 1 as 0 is reserved for the clone of the entry point.
    let hnsw_label_to_rox_id = index
      .labels()
      .enumerate()
      .map(|(id, l)| (l, id + 1))
      .collect::<HashMap<_, _>>();
    let mut vectors = index
      .labels()
      .map(|hnsw_label| {
        (
          hnsw_label_to_rox_id[&hnsw_label],
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
          .map(|hnsw_label| hnsw_label_to_rox_id[&hnsw_label])
          .collect_vec();
        let rox_id = hnsw_label_to_rox_id[&label];
        (rox_id, neighbors)
      })
      .collect::<HashMap<_, _>>();

    // TODO Copy deleted markers.
    let mut txn = DbTransaction::new();
    txn.write_cfg(&cfg);
    // Offset by 1 as 0 is an additional vector, the clone of the entry point.
    txn.write_count(vectors.len() + 1);
    txn.write_next_id(vectors.len() + 1);
    // Write internal entry point clone.
    {
      let entry_label = index.entry_label();
      let entry_id = hnsw_label_to_rox_id[&entry_label];
      txn.write_node(0, &NodeData {
        neighbors: graph[&entry_id].clone(),
        vector: Array1::from(vectors[&entry_id].clone()).mapv(|x| f16::from_f32(x)),
      });
    };
    for (label, id) in hnsw_label_to_rox_id {
      let key = label.to_string();
      txn.write_id(&key, id);
      txn.write_key(id, &key);
      txn.write_node(id, &NodeData {
        neighbors: graph.remove(&id).unwrap(),
        vector: Array1::from(vectors.remove(&id).unwrap()).mapv(|x| f16::from_f32(x)),
      });
    }
    txn.commit(&db).await;
    db.flush().await;
    drop(db);
    tracing::info!("all done!");
  }
}
