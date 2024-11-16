use crate::common::Dtype;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use dashmap::DashMap;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ordered_float::OrderedFloat;

pub struct BruteForceIndex<T: Dtype> {
  id_to_point: DashMap<Id, Array1<T>>,
  metric: Metric<T>,
}

impl<T: Dtype> BruteForceIndex<T> {
  pub fn new(metric: Metric<T>) -> Self {
    Self {
      id_to_point: DashMap::new(),
      metric,
    }
  }

  pub fn insert(&self, id: Id, vec: Array1<T>) {
    self.id_to_point.insert(id, vec);
  }

  pub fn query(&self, q: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    self
      .id_to_point
      .iter()
      .map(|e| PointDist {
        id: *e.key(),
        dist: (self.metric)(&e.view(), q),
      })
      .sorted_unstable_by_key(|e| OrderedFloat(e.dist))
      .take(k)
      .collect()
  }

  pub fn vectors(&self) -> &DashMap<Id, Array1<T>> {
    &self.id_to_point
  }

  pub fn len(&self) -> usize {
    self.id_to_point.len()
  }
}
