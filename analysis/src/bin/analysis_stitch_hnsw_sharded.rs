use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use dashmap::DashMap;
use hnswlib_rs::HnswIndex;
use itertools::Itertools;
use libroxanne::common::metric_euclidean;
use libroxanne::common::Id;
use libroxanne::common::PointDist;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::OptimizeMetrics;
use libroxanne::vamana::Vamana;
use libroxanne::vamana::VamanaDatastore;
use libroxanne::vamana::VamanaParams;
use ndarray::Array1;
use ordered_float::OrderedFloat;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use roxanne_analysis::analyse_index;
use roxanne_analysis::eval;
use roxanne_analysis::read_vectors;
use std::cmp::Reverse;
use std::fs;
use std::fs::File;
use std::sync::Arc;

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
  medoid: Id,
  node_to_level: DashMap<Id, usize>,
  nodes_by_level: DashMap<usize, Vec<Id>>,
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
    .nodes_by_level
    .iter()
    .sorted_unstable_by_key(|e| Reverse(*e.key()))
  {
    let level = *ent.key();
    // Don't reinsert all of level 0, as that's basically all of the graph.
    if level == 0 {
      continue;
    };
    let nodes = ent.value().clone();
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
fn strategy_stitch_cliques(
  ctx: &Ctx,
  ds: InMemoryVamana<f32>,
  params: VamanaParams,
) -> Vamana<f32, InMemoryVamana<f32>> {
  let index = Vamana::new(ds, metric_euclidean, ctx.medoid, params);

  let fast_greedy_k1_search =
    |graph: &HashMap<Id, Vec<Id>>, whitelist: &HashSet<Id>, query: Id| -> Option<Id> {
      let start = *graph
        .keys()
        .filter(|id| whitelist.contains(id))
        .choose(&mut thread_rng())?;
      let calc_dist = |a: Id, b: Id| {
        metric_euclidean(
          &index.datastore().get_point(a).unwrap().view(),
          &index.datastore().get_point(b).unwrap().view(),
        )
      };
      let mut cur = PointDist {
        id: start,
        dist: calc_dist(start, query),
      };
      let mut seen = HashSet::<Id>::new();
      loop {
        seen.insert(cur.id);
        // TODO Normally filtered nodes (i.e. `whitelist` in this case) must still be traversed, and is simply ignored in the final result, as actually ignoring them like they're deleted may be traversing a broken graph.
        let new = graph[&cur.id]
          .par_iter()
          .filter(|n| whitelist.contains(n) && !seen.contains(n))
          .map(|&n| PointDist {
            id: n,
            dist: calc_dist(cur.id, n),
          })
          .min_by_key(|e| OrderedFloat(e.dist));
        let Some(new) = new else {
          return Some(cur.id);
        };
        if cur.dist < new.dist {
          return Some(cur.id);
        };
        cur = new;
      }
    };

  // We want to go by level as nodes on different levels have different edge length and count distributions, which is probably ideal to preserve.
  for ent in ctx
    .nodes_by_level
    .iter()
    .sorted_unstable_by_key(|e| Reverse(*e.key()))
  {
    let level = *ent.key();
    let nodes = ent.value();
    let graphs = ctx
      .shards
      .iter()
      .map(|hnsw| {
        let mut graph = HashMap::<Id, Vec<Id>>::new();
        for &label in nodes.iter() {
          if hnsw.has_label(label) {
            graph.insert(label, hnsw.get_level_neighbors(label, level).collect());
          }
        }
        graph
      })
      .collect_vec();
    let mut available = nodes.iter().cloned().collect::<HashSet<_>>();
    while let Some(&base_node) = available.iter().next() {
      let mut nodes = Vec::new();
      let mut out_neighbors = Vec::<Id>::new();
      for g in graphs.iter() {
        let node = if g.contains_key(&base_node) {
          base_node
        } else {
          let Some(n) = fast_greedy_k1_search(g, &available, base_node) else {
            continue;
          };
          n
        };
        assert!(available.remove(&node));
        out_neighbors.extend(index.datastore().graph().get(&node).unwrap().iter());
        nodes.push(node);
      }
      for id in nodes {
        let mut n = index.datastore().graph().get_mut(&id).unwrap();
        n.clear();
        n.extend(out_neighbors.iter());
      }
    }

    let e = eval(&index, &ctx.queries, &ctx.ground_truths);
    println!(
      "[Level {level} with {} nodes] Correct: {:.2}% ({}/{})",
      nodes.len(),
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
  let nodes_by_level = DashMap::<usize, Vec<Id>>::new();
  let node_to_level = DashMap::<Id, usize>::new();
  let adj_list = DashMap::<Id, HashSet<Id>>::new();
  let id_to_point = DashMap::<Id, Array1<f32>>::new();
  hnsws.par_iter().for_each(|hnsw| {
    for id in hnsw.labels() {
      let level = hnsw.get_node_level(id);
      nodes_by_level.entry(level).or_default().push(id);
      node_to_level.insert(id, level);
      adj_list.insert(id, hnsw.get_merged_neighbors(id, 0));
      id_to_point.insert(id, Array1::from_vec(hnsw.get_data_by_label(id)));
    }
  });
  let entrypoints = hnsws.iter().map(|hnsw| hnsw.entry_label()).collect_vec();
  for &id in entrypoints.iter() {
    for &neighbor in entrypoints.iter() {
      if id != neighbor {
        adj_list.get_mut(&id).unwrap().insert(neighbor);
      }
    }
  }
  let ctx = Ctx {
    ground_truths,
    k,
    medoid: hnsws[0].entry_label(),
    node_to_level,
    nodes_by_level,
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

  println!("strategy_stitch_cliques");
  println!("============");
  let index = strategy_stitch_cliques(&ctx, ds.clone(), params.clone());
  println!();

  analyse_index("hnsw-sharded", &index);
}
