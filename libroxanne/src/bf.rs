use crate::common::Dtype;
use crate::common::DtypeCalc;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use crate::search::Points;
use crate::search::Query;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ordered_float::OrderedFloat;

pub struct BruteForceIndex<T: Dtype, C: DtypeCalc> {
  id_to_point: DashMap<Id, Array1<T>>,
  metric: Metric<C>,
}

impl<T: Dtype, C: DtypeCalc> BruteForceIndex<T, C> {
  pub fn new(metric: Metric<C>) -> Self {
    Self {
      id_to_point: DashMap::new(),
      metric,
    }
  }

  pub fn insert(&self, id: Id, vec: Array1<T>) {
    self.id_to_point.insert(id, vec);
  }

  pub fn delete(&self, id: Id) {
    self.id_to_point.remove(&id);
  }

  pub fn query(&self, q: &ArrayView1<T>, k: usize, filter: impl Fn(Id) -> bool) -> Vec<PointDist> {
    self
      .id_to_point
      .iter()
      .filter(|e| filter(*e.key()))
      .map(|e| PointDist {
        id: *e.key(),
        dist: self.dist3(&e.view(), Query::Vec(q)),
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

impl<T: Dtype, C: DtypeCalc> Points<T, C> for BruteForceIndex<T, C> {
  type Point<'a>
    = Ref<'a, Id, Array1<T>>
  where
    Self: 'a;

  fn metric(&self) -> Metric<C> {
    self.metric
  }

  fn precomputed_dists(&self) -> Option<&crate::common::PrecomputedDists> {
    None
  }

  fn get_point<'a>(&'a self, id: Id) -> Self::Point<'a> {
    self.id_to_point.get(&id).unwrap()
  }
}
