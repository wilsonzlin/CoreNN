use ahash::HashMap;
use clap::Args;
use hnswlib_rs::HnswIndexView;
use itertools::Itertools;
use libcorenn::cfg::Cfg;
use libcorenn::metric::StdMetric;
use libcorenn::store::schema::DbNodeData;
use libcorenn::store::schema::ID_TO_KEY;
use libcorenn::store::schema::KEY_TO_ID;
use libcorenn::store::schema::NODE;
use libcorenn::CoreNN;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::read;

#[derive(Args)]
pub struct MigrateHnswArgs {
  /// Path to a HNSW index file.
  #[arg()]
  path: PathBuf,

  /// Output directory to write CoreNN index to.
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
    let hnsw_raw = read(&self.path).await.unwrap();
    let index = HnswIndexView::load(self.dim, &hnsw_raw).unwrap();

    let cfg = Cfg {
      dim: self.dim,
      max_add_edges: index.m,
      max_edges: index.m,
      metric: self.metric,
      query_search_list_cap: index.ef,
      update_search_list_cap: index.ef_construction,
      ..Default::default()
    };

    // Make sure database can be created before we do long expensive work.
    let corenn = CoreNN::create(self.out, cfg);
    let db = corenn.internal_db();
    tracing::info!("created database");

    // In HNSW, "labels" are the external ID that the builder has defined for each vector.
    // To port to CoreNN, they will become the keys. We'll assign each a new internal CoreNN ID.
    // We offset by 1 as 0 is reserved for the clone of the entry point.
    let hnsw_label_to_corenn_id = index
      .labels()
      .enumerate()
      .map(|(id, l)| (l, id + 1))
      .collect::<HashMap<_, _>>();
    let mut vectors = index
      .labels()
      .map(|hnsw_label| {
        (
          hnsw_label_to_corenn_id[&hnsw_label],
          index.get_data_by_label(hnsw_label).unwrap().to_vec(),
        )
      })
      .collect::<HashMap<_, _>>();
    let mut graph = index
      .labels()
      .map(|label| {
        let neighbors = index
          .get_merged_neighbors(label, 0)
          .unwrap()
          .into_iter()
          .map(|hnsw_label| hnsw_label_to_corenn_id[&hnsw_label])
          .collect_vec();
        let corenn_id = hnsw_label_to_corenn_id[&label];
        (corenn_id, neighbors)
      })
      .collect::<HashMap<_, _>>();

    // TODO Copy deleted markers.
    // Offset by 1 as 0 is an additional vector, the clone of the entry point.
    // Write internal entry point clone.
    {
      let entry_label = index.entry_label().unwrap();
      let entry_id = hnsw_label_to_corenn_id[&entry_label];
      NODE.put(db, 0, DbNodeData {
        version: 0,
        neighbors: graph[&entry_id].clone(),
        // TODO Allow configuring dtype.
        vector: Arc::new(vectors[&entry_id].clone().into()),
      });
    };
    for (label, id) in hnsw_label_to_corenn_id {
      let key = label.to_string();
      KEY_TO_ID.put(db, &key, id);
      ID_TO_KEY.put(db, id, &key);
      NODE.put(db, id, DbNodeData {
        version: 0,
        neighbors: graph.remove(&id).unwrap(),
        vector: Arc::new(vectors.remove(&id).unwrap().into()),
      });
    }
    drop(corenn);
    tracing::info!("all done!");
  }
}
