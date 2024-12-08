use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use bytemuck::cast_slice;
use clap::Parser;
use libroxanne::common::Id;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to FastANN index.
  #[arg()]
  path: PathBuf,

  #[arg(long)]
  m: usize,

  #[arg(long)]
  k: usize,
}

#[allow(unused)]
#[derive(Deserialize)]
struct FastANNIndexLevelGraph {
  level: usize,
  #[serde(with = "serde_bytes")]
  nodes: Vec<u8>,
  #[serde(with = "serde_bytes")]
  neighbors_for_each_node: Vec<u8>,
}

#[allow(unused)]
#[derive(Deserialize)]
struct FastANNIndex {
  #[serde(with = "serde_bytes")]
  l0_graph: Vec<u8>,
  level_graphs: Vec<FastANNIndexLevelGraph>,
  entry_node: usize,
}

fn main() {
  let ds = Dataset::init();

  let metric = ds.info.metric.get_fn::<f32>();
  let vecs = ds.read_vectors();

  let args = Args::parse();

  let out_dir = format!("fastann-{}M-{}k", args.m, args.k);
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  let idx: FastANNIndex = rmp_serde::from_slice(&std::fs::read(args.path).unwrap()).unwrap();
  let mut level_graphs = HashMap::<usize, HashMap<Id, Vec<Id>>>::new();
  let mut graph = HashMap::<Id, HashSet<Id>>::new();
  // Process level 0.
  {
    let mat: &[i32] = cast_slice(&idx.l0_graph);
    assert_eq!(mat.len() % args.m, 0);
    for (id, neighbors) in mat.chunks(args.m).enumerate() {
      for &n in neighbors {
        if n == -1 {
          continue;
        };
        let n = Id::try_from(n).unwrap();
        level_graphs.entry(0).or_default().entry(id).or_default().push(n);
        level_graphs.entry(0).or_default().entry(n).or_default().push(id);
        graph.entry(id).or_default().insert(n);
        graph.entry(n).or_default().insert(id);
      }
    }
  };
  // Process other levels.
  idx.level_graphs.into_iter().for_each(|l| {
    let nodes: &[u32] = cast_slice(&l.nodes);
    let neighbors_mat: &[i32] = cast_slice(&l.neighbors_for_each_node);
    assert_eq!(neighbors_mat.len(), nodes.len() * args.m);
    for (neighbors, &id) in neighbors_mat.chunks(args.m).zip(nodes) {
      let id = Id::try_from(id).unwrap();
      for &n in neighbors {
        if n == -1 {
          continue;
        }
        let n = Id::try_from(n).unwrap();
        level_graphs.entry(l.level).or_default().entry(id).or_default().push(n);
        level_graphs.entry(l.level).or_default().entry(n).or_default().push(id);
        graph.entry(id).or_default().insert(n);
        graph.entry(n).or_default().insert(id);
      }
    }
  });
  println!("Loaded index");

  let mut graph_dists_by_level = HashMap::<usize, HashMap<Id, HashMap<Id, f64>>>::new();
  for (&level, level_nodes) in level_graphs.iter() {
    let level_dists = graph_dists_by_level.entry(level).or_default();
    for (&id, neighbors) in level_nodes {
      let neighbor_dists = neighbors.iter().map(|&neighbor| (
        neighbor,
        metric(&vecs.row(id), &vecs.row(neighbor)),
      )).collect();
      level_dists.insert(id, neighbor_dists);
    }
  }
  println!("Calculated edge dists by level");
  fs::write(
    format!(
      "dataset/{}/out/{out_dir}/edge_dists_by_level.msgpack",
      ds.name
    ),
    rmp_serde::to_vec_named(&graph_dists_by_level).unwrap(),
  )
  .unwrap();
  println!("Exported edge dists by level");

  export_index(
    &ds,
    &out_dir,
    &graph.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect(),
    idx.entry_node,
  );
}
