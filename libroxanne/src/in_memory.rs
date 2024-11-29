use crate::common::metric_euclidean;
use crate::common::Dtype;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PrecomputedDists;
use crate::search::GreedySearchable;
use crate::search::GreedySearchableSync;
use crate::vamana::Vamana;
use crate::vamana::VamanaParams;
use crate::vamana::VamanaSync;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use ndarray::Array1;
use ordered_float::OrderedFloat;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::iter::zip;
use std::sync::Arc;

pub fn calc_approx_medoid<T: Dtype>(
  id_to_point: &DashMap<Id, Array1<T>>,
  metric: Metric<T>,
  sample_size: usize,
  precomputed_dists: Option<&PrecomputedDists>,
) -> Id {
  let mut rng = thread_rng();
  let sample_ids = id_to_point
    .iter()
    .map(|e| *e.key())
    .choose_multiple(&mut rng, sample_size);
  sample_ids
    .par_iter()
    .copied()
    .min_by_key(|&i| {
      OrderedFloat(
        sample_ids
          .par_iter()
          .map(|&j| {
            precomputed_dists.map(|pd| pd.get(i, j)).unwrap_or_else(|| {
              metric(
                &id_to_point.get(&i).unwrap().view(),
                &id_to_point.get(&j).unwrap().view(),
              )
            })
          })
          .sum::<f64>(),
      )
    })
    .unwrap()
}

pub fn random_r_regular_graph(ids: &[Id], degree_bound: usize) -> DashMap<Id, Vec<Id>> {
  let adj_list = DashMap::new();
  ids.par_iter().for_each(|&id| {
    let mut rng = thread_rng();
    let neighbors = ids
      .choose_multiple(&mut rng, degree_bound + 1) // Choose +1 in case we pick self.
      .cloned()
      .filter(|&oid| id != oid)
      .take(degree_bound)
      .collect::<Vec<_>>();
    adj_list.insert(id, neighbors);
  });
  adj_list
}

/// Constructed via InMemoryIndex::builder().
pub struct InMemoryIndexBuilder<'a, T: Dtype> {
  ids: Vec<Id>,
  vectors: Vec<Array1<T>>,
  metric: Metric<T>,
  // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
  medoid_sample_size: usize,
  precomputed_dists: Option<Arc<PrecomputedDists>>,
  on_progress: Option<Box<dyn Fn(usize) + 'a>>,
  params: VamanaParams,
}

impl<'a, T: Dtype> InMemoryIndexBuilder<'a, T> {
  pub fn degree_bound(mut self, degree_bound: usize) -> Self {
    self.params.degree_bound = degree_bound;
    self
  }

  pub fn distance_threshold(mut self, distance_threshold: f64) -> Self {
    self.params.distance_threshold = distance_threshold;
    self
  }

  pub fn medoid_sample_size(mut self, medoid_sample_size: usize) -> Self {
    self.medoid_sample_size = medoid_sample_size;
    self
  }

  pub fn metric(mut self, metric: Metric<T>) -> Self {
    self.metric = metric;
    self
  }

  pub fn on_progress(mut self, on_progress: impl Fn(usize) + 'a) -> Self {
    self.on_progress = Some(Box::new(on_progress));
    self
  }

  pub fn precomputed_dists(
    mut self,
    precomputed_dists: impl Into<Option<Arc<PrecomputedDists>>>,
  ) -> Self {
    self.precomputed_dists = precomputed_dists.into();
    self
  }

  pub fn update_batch_size(mut self, update_batch_size: usize) -> Self {
    self.params.update_batch_size = update_batch_size;
    self
  }

  pub fn update_search_list_cap(mut self, update_search_list_cap: usize) -> Self {
    self.params.update_search_list_cap = update_search_list_cap;
    self
  }

  // The paper mentions running two passes, one with a=1.0. If you find that it gives more accuracy, you can do so by building initially with a=1.0, then updating a=$target and running optimize() again.
  // We won't do this here by default as it's costly.
  pub fn build(self) -> InMemoryIndex<T> {
    let InMemoryIndexBuilder {
      ids,
      medoid_sample_size,
      metric,
      on_progress,
      params,
      precomputed_dists,
      vectors,
    } = self;

    let graph = Arc::new(random_r_regular_graph(&ids, params.degree_bound));
    let vectors = Arc::new(zip(ids.clone(), vectors).collect::<DashMap<_, _>>());

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = calc_approx_medoid(
      &vectors,
      metric,
      medoid_sample_size,
      precomputed_dists.as_ref().map(|pd| pd.as_ref()),
    );

    let index = InMemoryIndex {
      graph,
      params,
      vectors,
      medoid,
      metric,
      precomputed_dists,
    };

    index.optimize(ids.clone(), None, |completed, _metrics| {
      if let Some(op) = on_progress.as_ref() {
        op(completed);
      }
    });

    index
  }
}

#[derive(Clone)]
pub struct InMemoryIndex<T: Dtype> {
  /// This can be useful for serializing to disk. Usually the points are already stored elsewhere, so serializing the graph alone can save space. To deserialize, collect the points, deserialize this graph, and use InMemoryVamana::new.
  /// This can also be used to introspect the graph, e.g. for debugging, analysis, or research.
  // This is Arc so it can be easily shared with InMemoryIndexBlob for serialization.
  pub graph: Arc<DashMap<Id, Vec<Id>>>,
  // This is Arc so it can be easily shared with InMemoryIndexBlob for serialization.
  pub vectors: Arc<DashMap<Id, Array1<T>>>,
  pub medoid: Id,
  pub metric: Metric<T>,
  pub params: VamanaParams,
  pub precomputed_dists: Option<Arc<PrecomputedDists>>,
}

impl<T: Dtype> InMemoryIndex<T> {
  pub fn builder<'a>(ids: Vec<Id>, vectors: Vec<Array1<T>>) -> InMemoryIndexBuilder<'a, T> {
    InMemoryIndexBuilder {
      ids,
      medoid_sample_size: 10_000,
      metric: metric_euclidean,
      on_progress: None,
      precomputed_dists: None,
      vectors,
      params: VamanaParams {
        degree_bound: 80,
        distance_threshold: 1.1,
        update_batch_size: 128,
        update_search_list_cap: 160,
      },
    }
  }

  pub fn len(&self) -> usize {
    self.vectors.len()
  }
}

impl<T: Dtype> GreedySearchable<T> for InMemoryIndex<T> {
  type FullVec = Vec<T>;
  type Neighbors<'a> = Ref<'a, Id, Vec<Id>>;
  type Point<'a> = Ref<'a, Id, Array1<T>>;

  fn medoid(&self) -> Id {
    self.medoid
  }

  fn metric(&self) -> Metric<T> {
    self.metric
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    self.precomputed_dists.as_ref().map(|pd| pd.as_ref())
  }

  fn get_point<'a>(&'a self, id: Id) -> Self::Point<'a> {
    self.vectors.get(&id).unwrap()
  }
}

impl<T: Dtype> GreedySearchableSync<T> for InMemoryIndex<T> {
  fn get_out_neighbors_sync<'a>(&'a self, id: Id) -> (Self::Neighbors<'a>, Option<Self::FullVec>) {
    (self.graph.get(&id).unwrap(), None)
  }
}

impl<T: Dtype> Vamana<T> for InMemoryIndex<T> {
  fn params(&self) -> &VamanaParams {
    &self.params
  }
}

impl<T: Dtype> VamanaSync<T> for InMemoryIndex<T> {
  fn set_point(&self, id: Id, point: Array1<T>) {
    self.vectors.insert(id, point);
  }

  fn set_out_neighbors(&self, id: Id, neighbors: Vec<Id>) {
    self.graph.insert(id, neighbors);
  }
}
