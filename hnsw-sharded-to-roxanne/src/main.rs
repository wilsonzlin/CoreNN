use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use clap::Parser;
use dashmap::DashMap;
use hnswlib_rs::HnswIndex;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use libroxanne::db::Db;
use libroxanne::db::NodeData;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::find_shortest_spanning_tree;
use libroxanne_search::greedy_search_fast1;
use libroxanne_search::metric_euclidean;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use libroxanne_search::PointDist;
use libroxanne_search::StdMetric;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::cmp::Reverse;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to a text file containing paths to HNSW shards.
  #[arg()]
  paths: PathBuf,

  /// Output directory to write Roxanne index to.
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
  let metric = args.metric.get_fn::<f32>();

  // Make sure database can be created before we do long expensive work.
  let db = Db::create(&args.out);
  println!("Created database");

  // First phase: load each shard to ensure integrity and compute some things.
  // - Ensure all shard files exist and are intact before we start doing long intensive work.
  // - Ensure all IDs are unique across all shards.
  let paths_raw = std::fs::read_to_string(&args.paths).unwrap();
  let paths = paths_raw.lines().collect::<Vec<_>>();
  // Map from level to (shard path, nodes).
  let shard_nodes_by_level = DashMap::<usize, Vec<(String, Vec<Id>)>>::new();
  let total_n = AtomicUsize::new(0);
  let seen_ids = DashMap::<Id, String>::new();
  let pb = new_pb(paths.len());
  paths.par_iter().for_each(|path| {
    let index = load_hnsw(args.dim, path);
    let mut map = HashMap::<usize, Vec<Id>>::new();
    for id in index.labels() {
      if let Some(other) = seen_ids.insert(id, path.to_string()) {
        panic!("Duplicate ID {id} in {other} and {path}");
      };
      let level = index.get_node_level(id);
      map.entry(level).or_default().push(id);
    }
    for (level, nodes) in map {
      total_n.fetch_add(nodes.len(), Ordering::Relaxed);
      shard_nodes_by_level
        .entry(level)
        .or_default()
        .push((path.to_string(), nodes));
    }
    pb.inc(1);
  });
  let total_n = total_n.load(Ordering::Relaxed);
  pb.finish();
  println!("Verified shards");

  let pb = new_pb(total_n);
  let mut medoid: Option<Id> = None;
  // Map from base node ID to map from shard path to other shard node ID.
  let cliques = DashMap::<Id, DashMap<String, Id>>::new();
  // Map from base node ID to combined out-neighbors.
  // We don't need a set, as a node cannot have an out-neighbor that also exists in another shard.
  let clique_neighbors = DashMap::<Id, Vec<Id>>::new();
  let node_to_clique = DashMap::<Id, Id>::new();
  // We want to go level-by-level as nodes on different levels have different edge length and count distributions, which is probably ideal to preserve.
  for (level, mut shards) in shard_nodes_by_level
    .into_iter()
    .sorted_unstable_by_key(|e| e.0)
    .rev()
  {
    // Use the shard with the most nodes at the current level as the base, as otherwise there may be some leftover nodes in other shards and the code/algorithm gets messy if we want to ensure they get used too. (Shards may not be perfectly balanced.)
    // Recalculate base shard for every level as different levels may have different largest shards.
    shards.sort_unstable_by_key(|e| Reverse(e.1.len()));

    let (base_file, base_level_nodes) = &shards[0];
    let base = load_hnsw(args.dim, base_file);
    let base_graph = base.build_level_index(level, base_level_nodes);
    // We can't just pick the entry point as the start that may not exist on our level.
    let base_path = find_shortest_spanning_tree(&base_graph, metric, base_level_nodes[0]);
    if medoid.is_none() {
      medoid = Some(base.entry_label());
    };

    base_level_nodes.par_iter().for_each(|&id| {
      assert!(cliques.insert(id, DashMap::new()).is_none());
      assert!(clique_neighbors
        .insert(id, base.get_merged_neighbors(id, 0).into_iter().collect())
        .is_none());
      assert!(node_to_clique.insert(id, id).is_none());
    });
    pb.inc(base_level_nodes.len() as u64);

    for batch in (&shards[1..]).chunks(args.parallel - 1) {
      batch.into_par_iter().for_each(|(path, shard_level_nodes)| {
        let other = load_hnsw(args.dim, path);
        let other_graph = other.build_level_index(level, &shard_level_nodes);
        let mut available = shard_level_nodes.iter().cloned().collect::<HashSet<_>>();

        for (base_id_from, base_id) in base_path.iter().cloned() {
          let base_vec = base_graph.get_point(base_id);

          let other_node = cliques
            // If None: we're at the start.
            .get(&base_id_from)
            // If None: no eqivalent to `from` in this shard was found previously.
            .and_then(|c| c.get(path).map(|e| *e))
            // We'll just use any point still available as the start. If there's not even a point available, we'll have to skip this shard.
            // NOTE: We cannot just use the HNSW entry node as it isn't available on this level (unless we're at the top level).
            .or_else(|| available.iter().cloned().next())
            .and_then(|start| {
              greedy_search_fast1(
                &other_graph,
                &base_vec.view(),
                metric_euclidean,
                start,
                |n| available.contains(&n),
              )
            });
          let Some(PointDist { id: other_node, .. }) = other_node else {
            continue;
          };

          assert!(available.remove(&other_node));
          cliques
            .get_mut(&base_id)
            .unwrap()
            .insert(path.clone(), other_node);
          clique_neighbors
            .get_mut(&base_id)
            .unwrap()
            .extend(other.get_merged_neighbors(other_node, 0));
          assert!(node_to_clique.insert(other_node, base_id).is_none());
          pb.inc(1);
        }
      });
    }
  }
  let nodes_touched = pb.position();
  pb.finish();
  println!("Processed shards: {nodes_touched} of {total_n} nodes updated");

  // Allow custom params for new super graph that will likely be stored on disk, which means the params might be different from a single HNSW shard's (so don't just copy existing).
  let cfg = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: args.distance_threshold,
    query_search_list_cap: args.query_search_list_cap,
    update_batch_size: args.update_batch_size,
    update_search_list_cap: args.update_search_list_cap,
  };

  db.write_cfg(&cfg);
  db.write_dim(args.dim);
  db.write_medoid(medoid.unwrap());
  db.write_metric(args.metric);

  let pb = new_pb(total_n);
  for path_batch in paths.chunks(args.parallel) {
    path_batch.into_par_iter().for_each(|path| {
      let index = load_hnsw(args.dim, path);
      // Collect to Vec so we can use into_par_iter, which is much faster than par_bridge.
      index.labels().collect_vec().into_par_iter().for_each(|id| {
        let neighbors = match node_to_clique.get(&id) {
          Some(clique_id) => clique_neighbors.get(&clique_id).unwrap().clone(),
          // This node was not matched to any clique, so just use existing neighbors.
          None => index.get_merged_neighbors(id, 0).into_iter().collect_vec(),
        };

        let node_data = NodeData {
          neighbors,
          vector: index.get_data_by_label(id),
        };

        db.write_node(id, &node_data);
        pb.inc(1);
      });
    });
  }
  pb.finish();
  println!("Finalizing database");

  db.flush();
  drop(db);
  println!("All done!");
}
