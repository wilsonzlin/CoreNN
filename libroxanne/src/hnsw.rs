use crate::common::Id;
use crate::common::Metric;
use crate::common::PrecomputedDists;
use crate::search::GreedySearchable;
use ahash::HashMap;
use hnswlib_rs::HnswIndex;
use hnswlib_rs::LabelType;
use itertools::Itertools;
use ndarray::Array1;

pub struct HnswLevelIndex<'h> {
  hnsw: &'h HnswIndex,
  metric: Metric<f32>,
  graph: HashMap<LabelType, Vec<LabelType>>,
}

impl<'h> HnswLevelIndex<'h> {
  pub fn new(
    hnsw: &'h HnswIndex,
    metric: Metric<f32>,
    level: usize,
    level_nodes: &[LabelType],
  ) -> Self {
    let graph = level_nodes
      .iter()
      .map(|&id| {
        (
          id,
          hnsw
            .get_level_neighbors(id, level)
            // We filter out edges to higher levels as they won't exist in this level's graph we're building.
            // TODO Is removing edges (and degrading the graph) better than including those higher-level nodes (i.e. flattening the graph up to this level)? This is still safe as those won't be picked due to being visited already. Measure accuracy and performance.
            .filter(|&n| hnsw.get_node_level(n) == level)
            .collect_vec(),
        )
      })
      .collect::<HashMap<_, _>>();
    Self {
      hnsw,
      metric,
      graph,
    }
  }

  pub fn base(&self) -> &'h HnswIndex {
    &self.hnsw
  }

  pub fn ids(&self) -> impl Iterator<Item = LabelType> + '_ {
    self.graph.keys().cloned()
  }
}

impl<'h, 'a> GreedySearchable<'a, f32> for HnswLevelIndex<'h> {
  type FullVec = Array1<f32>;
  type Neighbors = &'a Vec<LabelType>;
  type Point = Array1<f32>;

  fn medoid(&self) -> Id {
    self.hnsw.entry_label()
  }

  fn metric(&self) -> Metric<f32> {
    self.metric
  }

  fn get_point(&'a self, id: Id) -> Self::Point {
    Array1::from_vec(self.hnsw.get_data_by_label(id))
  }

  fn get_out_neighbors(&'a self, id: Id) -> (Self::Neighbors, Option<Self::FullVec>) {
    (self.graph.get(&id).unwrap(), None)
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    None
  }
}
