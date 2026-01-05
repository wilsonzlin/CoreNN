use ahash::HashMap;
use clap::Args;
use hnswlib_rs::legacy::load_hnswlib;
use hnswlib_rs::Cosine;
use hnswlib_rs::InnerProduct;
use hnswlib_rs::Metric;
use hnswlib_rs::VectorStore;
use hnswlib_rs::L2;
use hnswlib_rs::vector::Dense;
use libcorenn::cfg::Cfg;
use libcorenn::metric::StdMetric;
use libcorenn::store::schema::DbNodeData;
use libcorenn::store::schema::ADD_EDGES;
use libcorenn::store::schema::DELETED;
use libcorenn::store::schema::ID_TO_KEY;
use libcorenn::store::schema::KEY_TO_ID;
use libcorenn::store::schema::NODE;
use libcorenn::CoreNN;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::read;

#[derive(Clone)]
enum LegacyMetric {
  L2(L2),
  InnerProduct(InnerProduct),
  Cosine(Cosine),
}

impl Metric for LegacyMetric {
  type Family = Dense<f32>;

  fn distance<'a, 'b>(&self, a: &'a [f32], b: &'b [f32]) -> f32 {
    match self {
      LegacyMetric::L2(m) => m.distance(a, b),
      LegacyMetric::InnerProduct(m) => m.distance(a, b),
      LegacyMetric::Cosine(m) => m.distance(a, b),
    }
  }
}

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
    let metric = match self.metric {
      StdMetric::L2Sq => LegacyMetric::L2(L2::new()),
      StdMetric::Cosine => LegacyMetric::Cosine(Cosine::new()),
      StdMetric::InnerProduct => LegacyMetric::InnerProduct(InnerProduct::new()),
    };
    let (graph, vectors) = load_hnswlib(metric, self.dim, &hnsw_raw).unwrap();

    let cfg = Cfg {
      dim: self.dim,
      max_add_edges: graph.m(),
      max_edges: graph.m(),
      metric: self.metric,
      query_search_list_cap: graph.ef_search(),
      update_search_list_cap: graph.ef_construction(),
      ..Default::default()
    };

    // Make sure database can be created before we do long expensive work.
    let corenn = CoreNN::create(self.out, cfg);
    let db = corenn.internal_db();
    tracing::info!("created database");

    // In HNSW, "labels" are the external ID that the builder has defined for each vector.
    // To port to CoreNN, they will become the keys. We'll assign each a new internal CoreNN ID.
    // We offset by 1 as 0 is reserved for the clone of the entry point.
    let labels = graph.keys();
    let hnsw_label_to_corenn_id = labels
      .iter()
      .copied()
      .enumerate()
      .map(|(id, l)| (l, id + 1))
      .collect::<HashMap<_, _>>();

    // Offset by 1 as 0 is an additional vector, the clone of the entry point.
    // Write internal entry point clone.
    {
      let entry_label = graph.entry_key().unwrap();
      let entry_node = graph.node_id(&entry_label).unwrap();
      let entry_id = hnsw_label_to_corenn_id[&entry_label];
      let mut neighbors = graph
        .merged_neighbors(&entry_label, 0)
        .unwrap()
        .into_iter()
        .map(|hnsw_label| hnsw_label_to_corenn_id[&hnsw_label])
        .collect::<Vec<_>>();
      if !neighbors.contains(&entry_id) {
        neighbors.push(entry_id);
      }
      NODE.put(db, 0, DbNodeData {
        version: 0,
        neighbors,
        // TODO Allow configuring dtype.
        vector: Arc::new(vectors.vector(entry_node).unwrap().to_vec().into()),
      });
      ADD_EDGES.put(db, 0, Vec::<usize>::new());
    };

    for label in labels {
      let id = hnsw_label_to_corenn_id[&label];
      let node = graph.node_id(&label).unwrap();
      let deleted = graph.is_deleted_key(&label).unwrap();
      if deleted {
        DELETED.put(db, id, ());
      } else {
        let key = label.to_string();
        KEY_TO_ID.put(db, &key, id);
        ID_TO_KEY.put(db, id, &key);
      }

      let neighbors = graph
        .merged_neighbors(&label, 0)
        .unwrap()
        .into_iter()
        .map(|hnsw_label| hnsw_label_to_corenn_id[&hnsw_label])
        .collect::<Vec<_>>();
      NODE.put(db, id, DbNodeData {
        version: 0,
        neighbors,
        vector: Arc::new(vectors.vector(node).unwrap().to_vec().into()),
      });
      ADD_EDGES.put(db, id, Vec::<usize>::new());
    }
    drop(corenn);
    tracing::info!("all done!");
  }
}
