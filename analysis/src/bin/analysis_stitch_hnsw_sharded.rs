use ahash::HashSet;
use ahash::HashSetExt;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use dashmap::DashMap;
use hnswlib_rs::HnswIndex;
use itertools::Itertools;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::OptimizeMetrics;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaParams;
use libroxanne_search::find_shortest_spanning_path;
use libroxanne_search::greedy_search_fast1;
use libroxanne_search::metric_euclidean;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use roxanne_analysis::analyse_index;
use roxanne_analysis::eval;
use roxanne_analysis::read_vectors;
use std::cmp::Reverse;
use std::fs;
use std::fs::File;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long, default_value_t = 40)]
  degree_bound: usize,

  #[arg(long, default_value_t = 1.33)]
  query_search_list_cap_mul: f64,

  #[arg(long, default_value_t = 4.0)]
  update_search_list_cap_mul: f64,
}

struct Ctx {
  ground_truths: Vec<(Id, Array1<u32>)>,
  k: usize,
  level_to_shard_to_nodes: DashMap<usize, DashMap<usize, Vec<Id>>>,
  medoid: Id,
  node_to_level: DashMap<Id, usize>,
  queries: Vec<(Id, Array1<f32>)>,
  shards: Vec<HnswIndex>,
}

// This is somewhat of an ablation test: to see if other more-sophisticated strategies are actually doing anything better.
fn baseline_build_from_scratch(
  args: &Args,
  ctx: &Ctx,
  ds: InMemoryVamana<f32>,
  params: VamanaParams,
) {
  let mut ids = ctx.node_to_level.iter().map(|e| *e.key()).collect_vec();
  ids.shuffle(&mut thread_rng());
  ds.graph().par_iter_mut().for_each(|mut e| {
    let neighbors = e.value_mut();
    neighbors.clear();
    neighbors.extend(ids.choose_multiple(&mut thread_rng(), args.degree_bound));
  });
  let index = Vamana::new(ds, metric_euclidean, ctx.medoid, params);

  let batch_size = 500;
  let mut cumulative_updated_nodes = HashSet::<Id>::new();
  for (i, batch) in ids.chunks(batch_size).enumerate() {
    let mut metrics = OptimizeMetrics::default();
    index.optimize(batch.to_vec(), Some(&mut metrics));
    cumulative_updated_nodes.extend(metrics.updated_nodes.iter().copied());
    let touched_msg = metrics
      .updated_nodes
      .iter()
      .into_group_map_by(|e| *ctx.node_to_level.get(e).unwrap())
      .into_iter()
      .sorted_unstable_by_key(|e| e.0)
      .map(|(lvl, n)| format!("l{}={}", lvl, n.len()))
      .join(" ");
    let e = eval(&index, &ctx.queries, &ctx.ground_truths);
    println!("[Iteration {i}, {} nodes] Correct: {:.2}% ({}/{}) | Updated {} nodes ({} cumulatively): {touched_msg}", (i + 1) * batch_size, e.ratio() * 100.0, e.correct, e.total, metrics.updated_nodes.len(), cumulative_updated_nodes.len());
    if e.ratio() > 0.95 {
      break;
    }
  }
}

// This is somewhat of an ablation test: to see if other more-sophisticated strategies are actually doing anything better.
fn strategy_reinsert_randomly(ctx: &Ctx, ds: InMemoryVamana<f32>, params: VamanaParams) {
  let mut ids = ctx.node_to_level.iter().map(|e| *e.key()).collect_vec();
  ids.shuffle(&mut thread_rng());
  let index = Vamana::new(ds, metric_euclidean, ctx.medoid, params);

  let batch_size = 500;
  let mut cumulative_updated_nodes = HashSet::<Id>::new();
  for (i, batch) in ids.chunks(batch_size).enumerate() {
    let mut metrics = OptimizeMetrics::default();
    index.optimize(batch.to_vec(), Some(&mut metrics));
    cumulative_updated_nodes.extend(metrics.updated_nodes.iter().copied());
    let touched_msg = metrics
      .updated_nodes
      .iter()
      .into_group_map_by(|e| *ctx.node_to_level.get(e).unwrap())
      .into_iter()
      .sorted_unstable_by_key(|e| e.0)
      .map(|(lvl, n)| format!("l{}={}", lvl, n.len()))
      .join(" ");
    let e = eval(&index, &ctx.queries, &ctx.ground_truths);
    println!("[Iteration {i}, {} nodes] Correct: {:.2}% ({}/{}) | Updated {} nodes ({} cumulatively): {touched_msg}", (i + 1) * batch_size, e.ratio() * 100.0, e.correct, e.total, metrics.updated_nodes.len(), cumulative_updated_nodes.len());
    if e.ratio() > 0.95 {
      break;
    }
  }
}

fn strategy_reinsert_by_level(ctx: &Ctx, ds: InMemoryVamana<f32>, params: VamanaParams) {
  let index = Vamana::new(ds, metric_euclidean, ctx.medoid, params);

  let mut cumulative_updated_nodes = HashSet::<Id>::new();
  for ent in ctx
    .level_to_shard_to_nodes
    .iter()
    .sorted_unstable_by_key(|e| Reverse(*e.key()))
  {
    let level = *ent.key();
    // Don't reinsert all of level 0, as that's basically all of the graph.
    if level == 0 {
      continue;
    };
    let nodes = ent.iter().flat_map(|e| e.to_vec()).collect_vec();
    let n = nodes.len();
    let mut metrics = OptimizeMetrics::default();
    index.optimize(nodes, Some(&mut metrics));
    cumulative_updated_nodes.extend(metrics.updated_nodes.iter().copied());
    let touched_msg = metrics
      .updated_nodes
      .iter()
      .into_group_map_by(|e| *ctx.node_to_level.get(e).unwrap())
      .into_iter()
      .sorted_unstable_by_key(|e| e.0)
      .map(|(lvl, n)| format!("l{}={}", lvl, n.len()))
      .join(" ");
    let e = eval(&index, &ctx.queries, &ctx.ground_truths);
    println!("[Level {level} with {n} nodes] Correct: {:.2}% ({}/{}) | Updated {} nodes ({} cumulatively): {touched_msg}", e.ratio() * 100.0, e.correct, e.total, metrics.updated_nodes.len(), cumulative_updated_nodes.len());
  }
}

// This is just a fast approximation to something like k-means; it's not always accurate because it's unlikely that there are exactly N/S k-clusters of exactly S size, but hopefully it still works reasonably well.
// TODO Ablation studies:
// - Insert edges randomly across any levels.
// - Insert edges randomly within level.
// - Find random pair/clique (not closest) across shards and merge edges.
// - Stitch across all levels, not level-by-level.
// - Stitch subset of neighbors.
fn strategy_stitch_cliques(
  ctx: &Ctx,
  ds: InMemoryVamana<f32>,
  params: VamanaParams,
) -> Vamana<f32, InMemoryVamana<f32>> {
  let index = Vamana::new(ds, metric_euclidean, ctx.medoid, params);

  // We want to go by level as nodes on different levels have different edge length and count distributions, which is probably ideal to preserve.
  for ent in ctx
    .level_to_shard_to_nodes
    .iter()
    .sorted_unstable_by_key(|e| Reverse(*e.key()))
  {
    let level = *ent.key();
    let total_node_count = ent.iter().map(|e| e.len()).sum::<usize>();
    let graphs = ent
      .iter()
      .map(|e| {
        let hnsw = &ctx.shards[*e.key()];
        hnsw.build_level_graph(level, e.value())
      })
      .collect_vec();

    // Idea for faster finding of closest neighbor in another shard:
    // - Find closest node A' in shard 2 to base shard's start node A.
    // - Assumption: A ~= A', so neighbors(A) ~= neighbors(A').
    // - Traverse using Dijkstra, so we visit a node (B) only when we reach them by (hopefully) their closest neighbor (A) as other longer ways would not get queue-popped as soon.
    // - Let one neighbor of A be B. Query B from A' to find B'.
    // TODO Ablation study: just pick random node to start from for every query in another shard, instead of specifically the closest.

    // Map from base node ID to map from shard number to shard node ID.
    let cliques = DashMap::<Id, DashMap<usize, Id>>::new();
    // We can't just pick the entry point as the start that may not exist on our level.
    let base_path = find_shortest_spanning_path(
      &graphs[0],
      metric_euclidean,
      *graphs[0].level_graph().keys().next().unwrap(),
    );

    // In the base shard, the closest to `to` is from `from`.
    // Therefore, in every other shard, we query for the equivalent to `to` starting from the equivalent to the base `from` in the shard for faster convergence.
    graphs.par_iter().enumerate().skip(1).for_each(|(i, o)| {
      let mut available = o.level_graph().keys().cloned().collect::<HashSet<_>>();
      for (from, to) in base_path.iter().cloned() {
        let to_emb = graphs[0].get_point(to);

        // If None: we're at the start.
        cliques
          .get(&from)
          // If None: no eqivalent to `from` in this shard was found previously.
          .and_then(|c| c.get(&i).map(|e| *e))
          // We'll just use any point still available as the start. If there's not even a point available, we'll have to skip this shard.
          // NOTE: We cannot just use the HNSW entry node as it isn't available on this level (unless we're at the top level).
          .or_else(|| available.iter().cloned().next())
          .and_then(|start| {
            greedy_search_fast1(o, &to_emb.view(), metric_euclidean, start, |n| {
              available.contains(&n)
            })
          })
          .inspect(|&other_node| {
            cliques.entry(to).or_default().insert(i, other_node);
            assert!(available.remove(&other_node));
          });
      }
    });

    for (base_id, others) in cliques {
      // Get all neighbors, not just same-level ones.
      // No need for HashSet, as nodes can't exist across multiple shards.
      let mut combined_neighbors = graphs[0]
        .base()
        .get_merged_neighbors(base_id, 0)
        .into_iter()
        .collect_vec();
      for (i, o) in others.clone() {
        combined_neighbors.extend(graphs[i].base().get_merged_neighbors(o, 0));
      }

      // Update neighbors of all nodes in this clique.
      index
        .datastore()
        .graph()
        .insert(base_id, combined_neighbors.clone());
      for (_, o) in others {
        index
          .datastore()
          .graph()
          .insert(o, combined_neighbors.clone());
      }
    }

    let e = eval(&index, &ctx.queries, &ctx.ground_truths);
    println!(
      "[Level {level} with {total_node_count} nodes] Correct: {:.2}% ({}/{})",
      e.ratio() * 100.0,
      e.correct,
      e.total
    );
  }

  index
}

fn main() {
  let args = Args::parse();

  fs::create_dir_all("out/hnsw-sharded").unwrap();
  let shard_files = fs::read_dir("out/hnsw-sharded/indices")
    .unwrap()
    .map(|e| e.unwrap().path())
    .collect_vec();

  let queries = read_vectors("query.fvecs", LittleEndian::read_f32_into);
  let ground_truths = read_vectors("groundtruth.ivecs", LittleEndian::read_u32_into);
  let k = ground_truths[0].1.len();

  let hnsws = shard_files
    .par_iter()
    .map(|f| HnswIndex::load(128, File::open(f).unwrap()))
    .collect::<Vec<_>>();
  let level_to_shard_to_nodes = DashMap::<usize, DashMap<usize, Vec<Id>>>::new();
  let node_to_level = DashMap::<Id, usize>::new();
  let adj_list = DashMap::<Id, Vec<Id>>::new();
  let id_to_point = DashMap::<Id, Array1<f32>>::new();
  hnsws.par_iter().enumerate().for_each(|(shard_no, hnsw)| {
    for id in hnsw.labels() {
      let level = hnsw.get_node_level(id);
      level_to_shard_to_nodes
        .entry(level)
        .or_default()
        .entry(shard_no)
        .or_default()
        .push(id);
      node_to_level.insert(id, level);
      adj_list.insert(id, hnsw.get_merged_neighbors(id, 0).into_iter().collect());
      id_to_point.insert(id, Array1::from_vec(hnsw.get_data_by_label(id)));
    }
  });
  let entrypoints = hnsws.iter().map(|hnsw| hnsw.entry_label()).collect_vec();
  for &id in entrypoints.iter() {
    for &neighbor in entrypoints.iter() {
      if id != neighbor {
        adj_list.get_mut(&id).unwrap().push(neighbor);
      }
    }
  }
  let ctx = Ctx {
    ground_truths,
    k,
    level_to_shard_to_nodes,
    medoid: hnsws[0].entry_label(),
    node_to_level,
    queries,
    shards: hnsws,
  };

  // TODO Eval score should be based on distance, as it's more important for a near neighbor to be present than a far one.
  let ds = InMemoryVamana::new(adj_list, id_to_point);

  // TODO Investigate (degree_bound, insert_search_list_cap, query_search_list_cap) variations.
  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: 1.1,
    query_search_list_cap: (ctx.k as f64 * args.query_search_list_cap_mul) as usize,
    update_batch_size: num_cpus::get(),
    update_search_list_cap: (ctx.k as f64 * args.update_search_list_cap_mul) as usize,
  };

  println!("strategy_stitch_cliques");
  println!("============");
  let index = strategy_stitch_cliques(&ctx, ds.clone(), params.clone());
  println!();

  analyse_index("hnsw-sharded", &index);
  println!();

  // Other worse strategies, provided for reference. Ctrl+C (SIGINT) at this point if not useful.
  println!("baseline_build_from_scratch");
  println!("============");
  baseline_build_from_scratch(&args, &ctx, ds.clone(), params.clone());
  println!();

  println!("strategy_reinsert_by_level");
  println!("============");
  strategy_reinsert_by_level(&ctx, ds.clone(), params.clone());
  println!();

  println!("strategy_reinsert_randomly");
  println!("============");
  strategy_reinsert_randomly(&ctx, ds.clone(), params.clone());
  println!();
}
