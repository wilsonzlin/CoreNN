use crate::load_hnsw;
use crate::new_pb;
use ahash::HashSet;
use clap::Args;
use dashmap::DashMap;
use itertools::Itertools;
use libroxanne::common::Id;
use libroxanne::common::PointDist;
use libroxanne::common::StdMetric;
use libroxanne::db::Db;
use libroxanne::db::DbIndexMode;
use libroxanne::db::DbTransaction;
use libroxanne::db::NodeData;
use libroxanne::hnsw::HnswGraph;
use libroxanne::search::GreedySearchableSync;
use libroxanne::search::Points;
use libroxanne::search::Query;
use libroxanne::RoxanneDbDir;
use ndarray::ArrayView1;
use parking_lot::Mutex;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::cmp::Reverse;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Args)]
pub struct MigrateShardedHnswArgs {
  /// Path to a text file containing paths to HNSW shards.
  #[arg()]
  paths: PathBuf,

  /// Output directory to write Roxanne database to.
  #[arg(long)]
  out: PathBuf,

  /// How many shards (n) to compute in parallel; requires n shards in memory at one time and n threads.
  #[arg(long, default_value_t = 2)]
  parallel: usize,

  /// Dimensions of the vectors.
  #[arg(long)]
  dim: usize,

  /// Metric.
  #[arg(long)]
  metric: StdMetric,
}

impl MigrateShardedHnswArgs {
  // We don't perform PQ here, as it's expensive and not specifically related to this task; use the general `pqify` tool afterwards.
  pub async fn exec(self) {
    let metric = self.metric.get_fn::<f32>();

    let dir = RoxanneDbDir::new(self.out);

    // Make sure database can be created before we do long expensive work.
    let db = Db::open(dir.db()).await;
    tracing::info!("created database");

    // First phase: load each shard to ensure integrity and compute some things.
    // - Ensure all shard files exist and are intact before we start doing long intensive work.
    // - Ensure all IDs are unique across all shards.
    let paths_raw = std::fs::read_to_string(&self.paths).unwrap();
    let paths = paths_raw.lines().collect::<Vec<_>>();
    let total_n = AtomicUsize::new(0);
    let seen_ids = DashMap::<Id, String>::new();
    let pb = new_pb(paths.len());
    let shard_node_counts = paths
      .par_iter()
      .map(|path| {
        let index = load_hnsw(self.dim, path);
        total_n.fetch_add(index.cur_element_count, Ordering::Relaxed);
        for id in index.labels() {
          if let Some(other) = seen_ids.insert(id, path.to_string()) {
            panic!("Duplicate ID {id} in {other} and {path}");
          };
        }
        pb.inc(1);
        index.cur_element_count
      })
      .collect::<Vec<_>>();
    let total_n = total_n.load(Ordering::Relaxed);
    pb.finish();
    tracing::info!("verified shards");

    let pb = new_pb(total_n);
    // Map from base node ID to map from shard path to other shard node ID.
    let cliques = DashMap::<Id, DashMap<String, Id>>::new();
    // Map from base node ID to combined out-neighbors.
    // We don't need a set, as a node cannot have an out-neighbor that also exists in another shard.
    let clique_neighbors = DashMap::<Id, Vec<Id>>::new();
    let node_to_clique = DashMap::<Id, Id>::new();

    // Use the shard with the most nodes as the base, as otherwise there may be some leftover nodes in other shards and the code/algorithm gets messy if we want to ensure they get used too. (Shards may not be perfectly balanced.)
    let shards = (0..paths.len())
      .sorted_unstable_by_key(|&i| Reverse(shard_node_counts[i]))
      .map(|i| paths[i])
      .collect_vec();

    let base_file = shards[0];
    let base = load_hnsw(self.dim, base_file);
    let medoid = base.entry_label();
    let base_graph = HnswGraph::new(&base, metric, None);
    let base_path = base_graph.greedy_spanning_traversal(medoid);

    for id in base_graph.ids() {
      cliques.insert(id, DashMap::new());
      clique_neighbors.insert(id, base_graph.get_out_neighbors_sync(id).0.to_vec());
      node_to_clique.insert(id, id);
    }
    pb.inc(base.cur_element_count as u64);

    for batch in (&shards[1..]).chunks(self.parallel - 1) {
      batch.into_par_iter().for_each(|&path| {
        let other = load_hnsw(self.dim, path);
        let other_graph = HnswGraph::new(&other, metric, None);
        let mut available = other_graph.ids().collect::<HashSet<_>>();

        for (base_id_from, base_id) in base_path.iter().cloned() {
          let base_vec = ArrayView1::from(base_graph.get_point(base_id));

          let other_node = cliques
            // If None: we're at the start.
            .get(&base_id_from)
            // If None: no eqivalent to `from` in this shard was found previously.
            .and_then(|c| c.get(path).map(|e| *e))
            // We'll just use any point still available as the start. If there's not even a point available, we'll have to skip this shard.
            // NOTE: We cannot just use the HNSW entry node as it isn't available on this level (unless we're at the top level).
            .or_else(|| available.iter().cloned().next())
            .and_then(|start| {
              other_graph
                .greedy_search_fast1(Query::Vec(&base_vec), start, |n| available.contains(&n))
            });
          let Some(PointDist { id: other_node, .. }) = other_node else {
            continue;
          };

          assert!(available.remove(&other_node));
          cliques
            .get_mut(&base_id)
            .unwrap()
            .insert(path.to_string(), other_node);
          clique_neighbors
            .get_mut(&base_id)
            .unwrap()
            .extend(other_graph.get_out_neighbors_sync(other_node).0);
          assert!(node_to_clique.insert(other_node, base_id).is_none());
          pb.inc(1);
        }
      });
    }
    let nodes_touched = pb.position();
    pb.finish();
    tracing::info!(
      nodes_updated = nodes_touched,
      nodes_total = total_n,
      "processed shards and updated nodes"
    );

    let mut txn = DbTransaction::new();
    txn.write_index_mode(DbIndexMode::InMemory);
    txn.write_medoid(medoid);
    let txn = Mutex::new(txn);

    let pb = new_pb(total_n);
    for path_batch in paths.chunks(self.parallel) {
      path_batch.into_par_iter().for_each(|&path| {
        let index = load_hnsw(self.dim, path);
        // Collect to Vec so we can use into_par_iter, which is much faster than par_bridge.
        index.labels().collect_vec().into_par_iter().for_each(|id| {
          let neighbors = match node_to_clique.get(&id) {
            Some(clique_id) => clique_neighbors.get(&clique_id).unwrap().clone(),
            // This node was not matched to any clique, so just use existing neighbors.
            None => index.get_merged_neighbors(id, 0).into_iter().collect_vec(),
          };

          let node_data = NodeData {
            neighbors,
            vector: index.get_data_by_label(id).to_vec(),
          };

          txn.lock().write_node(id, &node_data);
          pb.inc(1);
        });
      });
    }
    pb.finish();
    tracing::info!("finalizing database");

    txn.into_inner().commit(&db).await;
    db.flush().await;
    drop(db);
    tracing::info!("all done!");
  }
}
