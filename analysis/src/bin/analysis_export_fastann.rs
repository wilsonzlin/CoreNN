use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use bytemuck::cast_slice;
use clap::Parser;
use dashmap::DashMap;
use itertools::Itertools;
use libroxanne::common::Id;
use roxanne_analysis::export_index;
use roxanne_analysis::Dataset;
use std::fs;

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

#[derive(Deserialize)]
struct FastANNIndexLevelGraph {
  level: usize,
  #[serde_with(serde_bytes)]
  nodes: Vec<u8>,
  #[serde_with(serde_bytes)]
  neighbors_for_each_node: Vec<u8>,
}

#[derive(Deserialize)]
struct FastANNIndex {
  #[serde_with(serde_bytes)]
  l0_graph: Vec<u8>,
  level_graphs: Vec<FastANNIndexLevelGraph>,
  entry_node: usize,
}

fn main() {
  let ds = Dataset::init();

  let args = Args::parse();

  let out_dir = format!("fastann-{}M-{}k", args.m, args.k);
  fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

  let idx: FastANNIndex = rmp_serde::from_slice(&std::fs::read(args.path).unwrap()).unwrap();
  let graph = HashMap::<Id, HashSet<Id>>::new();
  // Process level 0.
  {
    let mat: &[i32] = cast_slice(&idx.l0_graph);
    assert_eq!(mat.len() % args.m, 0);
    for (id, neighbors) in mat.chunks(args.m).enumerate() {
      for &n in neighbors {
        if n != -1 {
          graph.entry(id).or_default().insert(n as Id);
        }
      }
    }
  };
  // Process other levels.
  idx.level_graphs.into_iter().for_each(|l| {
    let nodes: &[u32] = cast_slice(&l.nodes);
    let neighbors_mat: &[i32] = cast_slice(&l.neighbors_for_each_node);
    assert_eq!(neighbors_mat.len(), nodes.len() * args.m);
    for (neighbors, &id) in neighbors_mat.chunks(args.m).zip(nodes) {
      for &n in neighbors {
        if n != -1 {
          graph.entry(id as Id).or_default().insert(n as Id);
        }
      }
    }
  });
  println!("Loaded index");

  export_index(
    &ds,
    &out_dir,
    &graph.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect(),
    idx.entry_node,
  );
}
