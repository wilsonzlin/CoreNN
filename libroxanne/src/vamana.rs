use ahash::HashMap;
use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashMap;
use itertools::Itertools;
use libroxanne_search::greedy_search;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use libroxanne_search::IdOrVec;
use libroxanne_search::Metric;
use libroxanne_search::PointDist;
use libroxanne_search::SearchMetrics;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use ordered_float::OrderedFloat;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;

pub fn calc_approx_medoid<T: Scalar + Send + Sync>(
  id_to_point: &DashMap<Id, Array1<T>>,
  metric: Metric<T>,
  sample_size: usize,
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
            metric(
              &id_to_point.get(&i).unwrap().view(),
              &id_to_point.get(&j).unwrap().view(),
            )
          })
          .sum::<f64>(),
      )
    })
    .unwrap()
}

// Return owned values:
// - We use DashMap for in-memory, so we can't return a ref while holding a lock in the map entry.
// - From disk, we copy the bytes.
pub trait VamanaDatastore<T: Scalar + Send + Sync>: GreedySearchable<T> + Send + Sync {
  fn set_point(&self, id: Id, point: Array1<T>);
  fn set_out_neighbors(&self, id: Id, neighbors: Vec<Id>);
}

#[derive(Clone, Default)]
pub struct InMemoryVamana<T: Scalar + Send + Sync> {
  adj_list: DashMap<Id, Vec<Id>>,
  id_to_point: DashMap<Id, Array1<T>>,
}

impl<T: Scalar + Send + Sync> GreedySearchable<T> for InMemoryVamana<T> {
  fn get_point(&self, id: Id) -> Array1<T> {
    self.id_to_point.get(&id).unwrap().clone()
  }

  fn get_out_neighbors(&self, id: Id) -> (Vec<Id>, Option<Array1<T>>) {
    (self.adj_list.get(&id).unwrap().clone(), None)
  }
}

impl<T: Scalar + Send + Sync> VamanaDatastore<T> for InMemoryVamana<T> {
  fn set_point(&self, id: Id, point: Array1<T>) {
    self.id_to_point.insert(id, point);
  }

  fn set_out_neighbors(&self, id: Id, neighbors: Vec<Id>) {
    self.adj_list.insert(id, neighbors);
  }
}

impl<T: Scalar + Send + Sync> InMemoryVamana<T> {
  pub fn new(adj_list: DashMap<Id, Vec<Id>>, id_to_point: DashMap<Id, Array1<T>>) -> Self {
    Self {
      adj_list,
      id_to_point,
    }
  }

  /// This can be useful for serializing to disk. Usually the points are already stored elsewhere, so serializing the graph alone can save space. To deserialize, collect the points, deserialize this graph, and use InMemoryVamana::new.
  /// This can also be used to introspect the graph, e.g. for debugging, analysis, or research.
  pub fn graph(&self) -> &DashMap<Id, Vec<Id>> {
    &self.adj_list
  }

  pub fn init_random_index(
    dataset: Vec<(Id, Array1<T>)>,
    metric: Metric<T>,
    params: VamanaParams,
    // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
    medoid_sample_size: usize,
    precomputed_dists: Option<Arc<(HashMap<Id, usize>, Vec<f16>)>>,
  ) -> Vamana<T, Self> {
    // Initialise to R-regular graph with random edges.
    let adj_list = DashMap::new();
    dataset.par_iter().for_each(|(id, _)| {
      let mut rng = thread_rng();
      let neighbors = dataset
        .choose_multiple(&mut rng, params.degree_bound + 1) // Choose +1 in case we pick self.
        .map(|e| e.0)
        .filter(|oid| id != oid)
        .take(params.degree_bound)
        .collect::<Vec<_>>();
      adj_list.insert(*id, neighbors);
    });
    let id_to_point = dataset.into_iter().collect::<DashMap<_, _>>();

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = calc_approx_medoid(&id_to_point, metric, medoid_sample_size);

    let ds = Self {
      adj_list,
      id_to_point,
    };
    let mut graph = Vamana::new(ds, metric, medoid, params);

    if let Some(pd) = precomputed_dists {
      graph.set_precomputed_dists(pd);
    };

    graph
  }

  pub fn build_index(
    dataset: Vec<(Id, Array1<T>)>,
    metric: Metric<T>,
    params: VamanaParams,
    // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
    medoid_sample_size: usize,
    precomputed_dists: Option<Arc<(HashMap<Id, usize>, Vec<f16>)>>,
    on_progress: impl Fn(usize, Option<&OptimizeMetrics>),
  ) -> Vamana<T, Self> {
    let mut ids_random = dataset.iter().map(|e| e.0).collect_vec();
    ids_random.shuffle(&mut thread_rng());

    let graph = Self::init_random_index(
      dataset,
      metric,
      params,
      medoid_sample_size,
      precomputed_dists,
    );

    graph.optimize(ids_random, None, on_progress);

    graph
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VamanaParams {
  pub query_search_list_cap: usize,
  pub update_search_list_cap: usize,
  // Corresponds to `α` in the DiskANN paper. Must be at least 1.
  pub distance_threshold: f64,
  // Corresponds to `R` in the DiskANN paper. The paper recommends at least log(N), where N is the number of points.
  pub degree_bound: usize,
  pub update_batch_size: usize,
  // Corresponds to W in the DiskANN paper, section 3.3 (DiskANN Beam Search).
  pub beam_width: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizeMetrics {
  pub updated_nodes: HashSet<Id>,
}

pub struct Vamana<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> {
  ds: DS,
  metric: Metric<T>,
  medoid: Id,
  params: VamanaParams,
  precomputed_dists: Option<Arc<(HashMap<Id, usize>, Vec<f16>)>>,
}

impl<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> Vamana<T, DS> {
  pub fn new(ds: DS, metric: Metric<T>, medoid: Id, params: VamanaParams) -> Self {
    Self {
      ds,
      metric,
      medoid,
      params,
      precomputed_dists: None,
    }
  }

  fn dist(&self, a: Id, b: Id) -> f64 {
    match self.precomputed_dists.as_ref() {
      None => (self.metric)(&self.ds.get_point(a).view(), &self.ds.get_point(b).view()),
      Some(pd) => {
        let (id_to_no, dists) = pd.as_ref();
        let n = id_to_no.len();
        let ia = id_to_no[&a];
        let ib = id_to_no[&b];
        dists[ia * n + ib] as f64
      }
    }
  }

  pub fn medoid(&self) -> Id {
    self.medoid
  }

  pub fn datastore(&self) -> &DS {
    &self.ds
  }

  pub fn params(&self) -> &VamanaParams {
    &self.params
  }

  pub fn params_mut(&mut self) -> &mut VamanaParams {
    &mut self.params
  }

  pub fn set_precomputed_dists(&mut self, precomputed_dists: Arc<(HashMap<Id, usize>, Vec<f16>)>) {
    self.precomputed_dists = Some(precomputed_dists);
  }

  fn greedy_search(
    &self,
    query: &ArrayView1<T>,
    k: usize,
    search_list_cap: usize,
    filter: impl Fn(PointDist) -> bool,
    metrics: Option<&mut SearchMetrics>,
    ground_truth: Option<&HashSet<Id>>,
  ) -> (Vec<PointDist>, HashSet<Id>) {
    let mut visited = HashSet::<Id>::new();
    let knn = greedy_search(
      &self.ds,
      IdOrVec::Vec(query),
      k,
      search_list_cap,
      self.params.beam_width,
      self.metric,
      self.medoid,
      filter,
      Some(&mut visited),
      self.precomputed_dists.clone(),
      metrics,
      ground_truth,
    );
    (knn, visited)
  }

  /// WARNING: `candidate_ids` must not contain the point itself.
  fn compute_robust_pruned(
    &self,
    node_id: Id,
    candidate_ids: impl IntoIterator<Item = Id>,
  ) -> Vec<Id> {
    let dist_thresh = self.params.distance_threshold;
    let degree_bound = self.params.degree_bound;

    // WARNING: Do not use into_par_iter as most callers are already threaded and this will cause extreme contention which slows down performance dramatically.
    let mut candidates = candidate_ids
      .into_iter()
      .map(|candidate_id| PointDist {
        id: candidate_id,
        dist: self.dist(node_id, candidate_id),
      })
      .sorted_unstable_by_key(|s| OrderedFloat(s.dist))
      .collect::<VecDeque<_>>();

    let mut new_neighbors = Vec::new();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some(PointDist { id: p_star, .. }) = candidates.pop_front() {
      new_neighbors.push(p_star);
      if new_neighbors.len() == degree_bound {
        break;
      }
      candidates.retain(|s| {
        let s_to_p = s.dist;
        let s_to_p_star = self.dist(p_star, s.id);
        s_to_p <= s_to_p_star * dist_thresh
      });
    }
    new_neighbors
  }

  // The point referenced by each ID should already be inserted into the DB.
  // This is used when inserting, but also during initialization, so this is a separate function from `insert`.
  // WARNING: The graph must not be mutated while this function is executing, but it is up to the caller to ensure this.
  /// WARNING: This is publicly exposed, but use this only if you know what you're doing. There are no guarantees of API stability.
  pub fn optimize(
    &self,
    mut ids: Vec<Id>,
    mut metrics: Option<&mut OptimizeMetrics>,
    on_progress: impl Fn(usize, Option<&OptimizeMetrics>),
  ) {
    // Shuffle to reduce chance of inserting around the same area in latent space.
    ids.shuffle(&mut thread_rng());
    let mut completed = 0;
    for batch in ids.chunks(self.params.update_batch_size) {
      #[derive(Default)]
      struct Update {
        // These two aren't the same and can't be merged, as otherwise we can't tell whether we are supposed to replace or merge with the existing out-neighbors.
        replacement_base: Option<Vec<Id>>,
        additional_edges: HashSet<Id>,
      }
      let updates = DashMap::<Id, Update>::new();

      batch.into_par_iter().for_each(|&id| {
        // TODO Delete if already exists.

        // Initial GreedySearch.
        let mut candidates = self
          .greedy_search(
            &self.ds.get_point(id).view(),
            1,
            self.params.update_search_list_cap,
            |_| true,
            None,
            None,
          )
          .1;

        // RobustPrune.
        // RobustPrune requires locking the graph node at this point; we're already holding the lock so we're good to go.
        for n in self.ds.get_out_neighbors(id).0 {
          candidates.insert(n);
        }
        // RobustPrune requires that the point itself is never in the candidate set.
        candidates.remove(&id);
        // It's tempting to do this at the end once only, in case other points in this batch will add an edge to us (which means another round of RobustPrune),
        // but that means `new_neighbors` will be a lot bigger (it'll just be the unpruned raw candidate set),
        // which means dirtying a lot more other nodes (and also adding a lot of poor edges), ultimately spending more compute.
        let new_neighbors = self.compute_robust_pruned(id, candidates);
        for &j in new_neighbors.iter() {
          updates.entry(j).or_default().additional_edges.insert(id);
        }
        updates.entry(id).or_default().replacement_base = Some(new_neighbors);
      });

      // Update dirty nodes in this batch.
      if let Some(m) = &mut metrics {
        m.updated_nodes.extend(updates.iter().map(|u| *u.key()));
      };
      updates.into_par_iter().for_each(|(id, u)| {
        let mut new_neighbors = u
          .replacement_base
          .unwrap_or_else(|| self.ds.get_out_neighbors(id).0);
        for j in u.additional_edges {
          if !new_neighbors.contains(&j) {
            new_neighbors.push(j);
          };
        }
        if new_neighbors.len() > self.params.degree_bound {
          new_neighbors = self.compute_robust_pruned(id, new_neighbors);
        };
        self.ds.set_out_neighbors(id, new_neighbors);
      });
      completed += batch.len();
      on_progress(completed, metrics.as_ref().map(|m| &**m));
    }
  }

  pub fn insert(&self, points: Vec<(Id, Array1<T>)>) {
    let ids = points.iter().map(|(id, _)| *id).collect_vec();
    points
      .into_par_iter()
      .for_each(|(id, point)| self.ds.set_point(id, point));
    self.optimize(ids, None, |_, _| {});
  }

  pub fn query_with_filter(
    &self,
    query: &ArrayView1<T>,
    k: usize,
    filter: impl Fn(PointDist) -> bool,
  ) -> Vec<PointDist> {
    self
      .greedy_search(
        query,
        k,
        self.params.query_search_list_cap,
        filter,
        None,
        None,
      )
      .0
  }

  pub fn query_with_metrics(
    &self,
    query: &ArrayView1<T>,
    k: usize,
    ground_truth: Option<&HashSet<Id>>,
  ) -> (Vec<PointDist>, SearchMetrics) {
    let mut metrics = SearchMetrics::default();
    let res = self
      .greedy_search(
        query,
        k,
        self.params.query_search_list_cap,
        |_| true,
        Some(&mut metrics),
        ground_truth,
      )
      .0;
    (res, metrics)
  }

  pub fn query(&self, query: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    self.query_with_filter(query, k, |_| true)
  }
}

#[cfg(test)]
mod tests {
  use super::VamanaParams;
  use crate::vamana::InMemoryVamana;
  use ahash::HashSet;
  use itertools::Itertools;
  use libroxanne_search::metric_euclidean;
  use ndarray::Array;
  use ndarray::Array1;
  use ndarray_rand::RandomExt;
  use ordered_float::OrderedFloat;
  use rand::distributions::Uniform;
  use std::iter::zip;

  #[test]
  fn test_vamana_512d() {
    let metric = metric_euclidean;
    const DIM: usize = 512;
    let n = 1000usize;
    let r = 12;
    let ids = (0..n).collect_vec();
    let k = 15;
    let search_list_cap = k * 2;

    fn gen_vec() -> Array1<f32> {
      Array::random((DIM,), Uniform::new(-10.0f32, 10.0f32))
    }

    let points = (0..n).map(|_| gen_vec()).collect_vec();
    let dataset = zip(ids.clone(), points.clone()).collect_vec();

    let vamana = InMemoryVamana::build_index(
      dataset,
      metric,
      VamanaParams {
        beam_width: 1,
        degree_bound: r,
        distance_threshold: 1.1,
        query_search_list_cap: search_list_cap,
        update_batch_size: 64,
        update_search_list_cap: search_list_cap,
      },
      10_000,
      None,
      |_, _| {},
    );

    // First, test ANN of every point.
    let mut correct = 0;
    for a in ids.iter().cloned() {
      let a_pt = &points[a];
      let truth = ids
        .iter()
        .cloned()
        .filter(|&b| b != a)
        .sorted_unstable_by_key(|&b| OrderedFloat(metric(&a_pt.view(), &points[b].view())))
        .take(k)
        .collect::<HashSet<_>>();
      let approx = vamana
        .query(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
        .into_iter()
        .map(|pd| pd.id)
        .filter(|&b| b != a)
        .take(k)
        .collect::<HashSet<_>>();
      correct += approx.intersection(&truth).count();
    }
    println!(
      "[512D Pairwise] Correct: {}/{} ({:.2}%)",
      correct,
      k * n,
      correct as f64 / (k * n) as f64 * 100.0
    );

    // Second, test ANN of a query.
    let query = gen_vec();
    let truth = ids
      .iter()
      .cloned()
      .sorted_unstable_by_key(|&id| OrderedFloat(metric(&query.view(), &points[id].view())))
      .take(k)
      .collect::<HashSet<_>>();
    let approx = vamana
      .query(&query.view(), k)
      .into_iter()
      .map(|pd| pd.id)
      .collect::<HashSet<_>>();
    let correct = approx.intersection(&truth).count();
    println!(
      "[512D Query] Correct: {}/{} ({:.2}%)",
      correct,
      k,
      correct as f64 / k as f64 * 100.0
    );
  }
}
