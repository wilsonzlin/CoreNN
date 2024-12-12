use crate::export_index;
use crate::read_graph_matrix;
use crate::Dataset;
use ahash::HashMap;
use ahash::HashMapExt;
use bytemuck::cast_slice;
use dashmap::DashMap;
use libroxanne::common::Id;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rmp_serde::from_slice;
use rmp_serde::to_vec_named;
use serde::Deserialize;
use std::cmp::min;
use std::fs::read;
use std::fs::read_to_string;
use std::fs::write;

#[allow(unused)]
#[derive(Deserialize)]
struct LevelGraph {
  level: usize,
  #[serde(with = "serde_bytes")]
  nodes: Vec<u8>,
  #[serde(with = "serde_bytes")]
  neighbors_for_each_node: Vec<u8>,
}

pub fn export_randinit(ds: &Dataset, out: &str, variant: &str, m: usize) {
  let metric = ds.info.metric.get_fn::<f32>();
  let vecs = ds.read_vectors();

  let graph = read_graph_matrix(format!("{out}/graph.mat"), (ds.info.n, m));

  let medoid = read_to_string(format!("{out}/medoid.txt"))
    .unwrap()
    .parse::<Id>()
    .unwrap();

  let level_graphs = DashMap::<usize, DashMap<Id, Vec<Id>>>::new();
  let level_graphs_raw: Vec<LevelGraph> =
    from_slice(&read(format!("{out}/level_graphs.msgpack")).unwrap()).unwrap();
  level_graphs_raw.into_par_iter().for_each(|l| {
    let nodes: &[u32] = cast_slice(&l.nodes);
    let neighbors_mat: &[u32] = cast_slice(&l.neighbors_for_each_node);
    assert_eq!(neighbors_mat.len(), nodes.len() * min(m, nodes.len()));
    for (neighbors, &id) in neighbors_mat.chunks(m).zip(nodes) {
      let id = Id::try_from(id).unwrap();
      for &n in neighbors {
        if n == i32::MAX as u32 {
          continue;
        }
        let n = Id::try_from(n).unwrap();
        level_graphs
          .entry(l.level)
          .or_default()
          .entry(id)
          .or_default()
          .push(n);
        graph.entry(id).or_default().push(n);
      }
    }
  });

  let mut graph_dists_by_level = HashMap::<usize, HashMap<Id, HashMap<Id, f64>>>::new();
  for (level, level_nodes) in level_graphs {
    let level_dists = graph_dists_by_level.entry(level).or_default();
    for (id, neighbors) in level_nodes {
      let neighbor_dists = neighbors
        .iter()
        .map(|&neighbor| (neighbor, metric(&vecs.row(id), &vecs.row(neighbor))))
        .collect();
      level_dists.insert(id, neighbor_dists);
    }
  }
  println!("Calculated edge dists by level");
  write(
    format!(
      "dataset/{}/out/{variant}/edge_dists_by_level.msgpack",
      ds.name
    ),
    to_vec_named(&graph_dists_by_level).unwrap(),
  )
  .unwrap();
  println!("Exported edge dists by level");

  export_index(&ds, &variant, &graph, medoid);
}
