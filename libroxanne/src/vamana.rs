use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashMap;
use dashmap::DashSet;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "$type")]
pub enum VamanaInstrumentationEvent<T: Scalar> {
  OptimizeBatchBegin {
    batch_ids: Vec<Id>,
  },
  GreedySearchIteration {
    query: Array1<T>,
    expanded: Vec<Id>,
    neighbors_of_expanded: Vec<Id>,
    new_unvisited: Vec<Id>,
    final_search_list_visited: Vec<Id>,
    final_search_list_unvisited: Vec<Id>,
  },
  RobustPruneBegin {
    node: Id,
    candidates: Vec<Id>,
  },
  RobustPruneIteration {
    node: Id,
    p_star: Id,
    final_candidates: Option<Vec<Id>>, // None on final iteration.
  },
}

pub type VamanaInstrumentation<T> = Box<dyn Fn(VamanaInstrumentationEvent<T>) + Send + Sync>;

// Return owned values:
// - We use DashMap for in-memory, so we can't return a ref while holding a lock in the map entry.
// - From disk, we copy the bytes.
pub trait VamanaDatastore<T: Scalar + Send + Sync>: Send + Sync {
  fn get_point(&self, id: Id) -> Option<Array1<T>>;
  fn set_point(&self, id: Id, point: Array1<T>);
  fn get_out_neighbors(&self, id: Id) -> Option<HashSet<Id>>;
  fn set_out_neighbors(&self, id: Id, neighbors: HashSet<Id>);
}

#[derive(Clone, Default)]
pub struct InMemoryVamana<T: Scalar + Send + Sync> {
  adj_list: DashMap<Id, HashSet<Id>>,
  id_to_point: DashMap<Id, Array1<T>>,
}

impl<T: Scalar + Send + Sync> VamanaDatastore<T> for InMemoryVamana<T> {
  fn get_point(&self, id: Id) -> Option<Array1<T>> {
    self.id_to_point.get(&id).map(|e| e.clone())
  }

  fn set_point(&self, id: Id, point: Array1<T>) {
    self.id_to_point.insert(id, point);
  }

  fn get_out_neighbors(&self, id: Id) -> Option<HashSet<Id>> {
    self.adj_list.get(&id).map(|e| e.clone())
  }

  fn set_out_neighbors(&self, id: Id, neighbors: HashSet<Id>) {
    self.adj_list.insert(id, neighbors);
  }
}

impl<T: Scalar + Send + Sync> InMemoryVamana<T> {
  pub fn new(adj_list: DashMap<Id, HashSet<Id>>, id_to_point: DashMap<Id, Array1<T>>) -> Self {
    Self {
      adj_list,
      id_to_point,
    }
  }

  /// This can be useful for serializing to disk. Usually the points are already stored elsewhere, so serializing the graph alone can save space. To deserialize, collect the points, deserialize this graph, and use InMemoryVamana::new.
  /// This can also be used to introspect the graph, e.g. for debugging, analysis, or research.
  pub fn graph(&self) -> &DashMap<Id, HashSet<Id>> {
    &self.adj_list
  }

  pub fn init_random_index(
    dataset: Vec<(Id, Array1<T>)>,
    metric: Metric<T>,
    params: VamanaParams,
    // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
    medoid_sample_size: usize,
    instrumentation: Option<VamanaInstrumentation<T>>,
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
        .collect::<HashSet<_>>();
      adj_list.insert(*id, neighbors);
    });
    let ids = dataset.iter().map(|e| e.0).collect_vec();
    let id_to_point = dataset.into_iter().collect::<DashMap<_, _>>();

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = {
      let mut rng = thread_rng();
      let sample_ids = ids
        .choose_multiple(&mut rng, medoid_sample_size)
        .copied()
        .collect_vec();
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
    };

    let ds = Self {
      adj_list,
      id_to_point,
    };
    let mut graph = Vamana::new(ds, metric, medoid, params);

    if let Some(instrumentation) = instrumentation {
      graph.set_instrumentation(instrumentation);
    };

    graph
  }

  pub fn build_index(
    dataset: Vec<(Id, Array1<T>)>,
    metric: Metric<T>,
    params: VamanaParams,
    // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
    medoid_sample_size: usize,
    instrumentation: Option<VamanaInstrumentation<T>>,
  ) -> Vamana<T, Self> {
    let mut ids_random = dataset.iter().map(|e| e.0).collect_vec();
    ids_random.shuffle(&mut thread_rng());

    let graph =
      Self::init_random_index(dataset, metric, params, medoid_sample_size, instrumentation);

    graph.optimize(ids_random, None);

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

#[derive(Clone, Copy)]
enum IdOrPoint<'a, 'b, T: Scalar> {
  Id(Id),
  Point(&'a ArrayView1<'b, T>),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SearchIterationMetrics {
  pub visited: usize,
  pub unvisited_dist_sum: f64,   // The sum of all unvisited nodes.
  pub unvisited_dist_mins: f64,  // The min. dist. of all unvisited nodes.
  pub dropped_candidates: usize, // Neighbors of an expanded node that were already visited or in the unvisited queue.
  pub new_candidates: usize,
  pub dropped_visited: usize,
  pub dropped_unvisited: usize,
  pub ground_truth_found: usize, // Cumulative.
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
  pub iterations: Vec<SearchIterationMetrics>,
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
  instrumentation: Option<VamanaInstrumentation<T>>,
}

impl<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> Vamana<T, DS> {
  pub fn new(ds: DS, metric: Metric<T>, medoid: Id, params: VamanaParams) -> Self {
    Self {
      ds,
      metric,
      medoid,
      params,
      instrumentation: None,
    }
  }

  fn dist(&self, a: Id, b: Id) -> f64 {
    self.dist2(a, IdOrPoint::Id(b))
  }

  fn dist2(&self, a: Id, b: IdOrPoint<T>) -> f64 {
    match b {
      IdOrPoint::Id(b) => {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        (self.metric)(
          &self.ds.get_point(a).unwrap().view(),
          &self.ds.get_point(b).unwrap().view(),
        )
      }
      IdOrPoint::Point(b) => {
        let a = self.ds.get_point(a).unwrap();
        (self.metric)(&a.view(), b)
      }
    }
  }

  pub fn set_instrumentation(&mut self, instrumentation: VamanaInstrumentation<T>) {
    self.instrumentation = Some(instrumentation);
  }

  fn inst(&self, evt_fn: impl FnOnce() -> VamanaInstrumentationEvent<T>) {
    if let Some(instrumentation) = &self.instrumentation {
      instrumentation(evt_fn());
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

  // DiskANN paper, Algorithm 1: GreedySearch.
  // Returns a pair: (closest points, visited node IDs).
  // Filtered nodes will be visited and expanded but not considered for the final set of neighbors.
  fn greedy_search(
    &self,
    query: IdOrPoint<T>,
    k: usize,
    search_list_cap: usize,
    filter: impl Fn(PointDist) -> bool,
    mut metrics: Option<&mut SearchMetrics>,
    ground_truth: Option<&HashSet<Id>>,
  ) -> (Vec<PointDist>, HashSet<Id>) {
    let start = self.medoid;
    assert!(
      search_list_cap >= k,
      "search list capacity must be greater than or equal to k"
    );

    // It's too inefficient to calculate L\V repeatedly.
    // Since we need both L (return value) and L\V (each iteration), we split L into V and ¬V.
    // For simplicity, we'll just allow both to reach `k` size, and do a final merge at the end. This doubles the memory requirements, but in reality `k` is often small enough that it's not a problem.
    // We also need `all_visited` as `l_visited` truncates to `k`, but we also want all visited points in the end.
    // L = l_visited + l_unvisited
    // `l_unvisited_set` is for members currently in l_unvisited, to avoid pushing duplicates. (NOTE: This is *not* the same as `all_visited`.) We don't need one for `l_visited` because we only push popped elements from `l_unvisited`, which we guarantee are unique (as previously mentioned).
    let mut l_unvisited_set = HashSet::new();
    let mut l_unvisited = VecDeque::<PointDist>::new(); // L \ V
    let mut l_visited = VecDeque::<PointDist>::new(); // V
    let mut all_visited = HashSet::new();
    l_unvisited.push_back(PointDist {
      id: start,
      dist: self.dist2(start, query),
    });
    let ground_truth_found = AtomicUsize::new(0);
    while !l_unvisited.is_empty() {
      let new_visited = (0..self.params.beam_width)
        .filter_map(|_| l_unvisited.pop_front())
        .collect::<Vec<_>>();
      let neighbors = DashSet::new();
      new_visited.par_iter().for_each(|p_star| {
        for j in self.ds.get_out_neighbors(p_star.id).unwrap_or_default() {
          neighbors.insert(j);
        }
      });
      // Move to visited section.
      all_visited.extend(new_visited.iter().map(|e| e.id));
      l_visited.extend(new_visited.iter().filter(|e| filter(**e)));
      l_visited
        .make_contiguous()
        .sort_unstable_by_key(|s| OrderedFloat(s.dist));

      let new_unvisited = neighbors
        .par_iter()
        .filter_map(|neighbor| {
          let neighbor = *neighbor;
          // We separate L out into V and not V, so we must manually ensure the property that l_visited and l_unvisited are disjoint.
          if all_visited.contains(&neighbor) {
            return None;
          };
          if l_unvisited_set.contains(&neighbor) {
            return None;
          };
          if ground_truth.is_some_and(|gt| gt.contains(&neighbor)) {
            ground_truth_found.fetch_add(1, Ordering::Relaxed);
          }
          Some(PointDist {
            id: neighbor,
            dist: self.dist2(neighbor, query),
          })
        })
        .collect::<VecDeque<_>>();
      l_unvisited_set.extend(new_unvisited.iter().map(|e| e.id));
      l_unvisited.extend(&new_unvisited);
      l_unvisited
        .make_contiguous()
        .sort_unstable_by_key(|s| OrderedFloat(s.dist));

      let mut dropped_unvisited = 0;
      let mut dropped_visited = 0;
      while l_unvisited.len() + l_visited.len() > search_list_cap {
        let (Some(u), Some(v)) = (l_unvisited.back(), l_visited.back()) else {
          break;
        };
        if u.dist >= v.dist {
          l_unvisited.pop_back();
          dropped_unvisited += 1;
        } else {
          l_visited.pop_back();
          dropped_visited += 1;
        }
      }

      if let Some(m) = &mut metrics {
        m.iterations.push(SearchIterationMetrics {
          dropped_candidates: neighbors.len() - new_unvisited.len(),
          dropped_unvisited,
          dropped_visited,
          ground_truth_found: ground_truth_found.load(Ordering::Relaxed),
          new_candidates: new_unvisited.len(),
          unvisited_dist_mins: l_unvisited.front().map(|n| n.dist).unwrap_or_default(),
          unvisited_dist_sum: l_unvisited.iter().map(|n| n.dist).sum(),
          visited: all_visited.len(),
        });
      }
      self.inst(|| VamanaInstrumentationEvent::GreedySearchIteration {
        query: match query {
          IdOrPoint::Id(id) => self.ds.get_point(id).unwrap(),
          IdOrPoint::Point(p) => p.to_owned(),
        },
        expanded: new_visited.iter().map(|e| e.id).collect(),
        neighbors_of_expanded: neighbors.into_iter().collect(),
        new_unvisited: new_unvisited.iter().map(|e| e.id).collect(),
        final_search_list_visited: l_visited.iter().map(|e| e.id).collect(),
        final_search_list_unvisited: l_unvisited.iter().map(|e| e.id).collect(),
      });
    }

    // Find the k closest points from both l_visited + l_unvisited (= L).
    let mut closest = Vec::new();
    while closest.len() < k {
      match (l_visited.pop_front(), l_unvisited.pop_front()) {
        (None, None) => break,
        (Some(v), None) | (None, Some(v)) => closest.push(v),
        (Some(a), Some(b)) => {
          if a.dist < b.dist {
            closest.push(a);
            closest.push(b);
          } else {
            closest.push(b);
            closest.push(a);
          };
        }
      };
    }
    // We may have exceeded k due to pushing both a and b in the last match arm.
    closest.truncate(k);
    (closest, all_visited)
  }

  /// WARNING: `candidate_ids` must not contain the point itself.
  fn compute_robust_pruned(&self, node_id: Id, candidate_ids: HashSet<Id>) -> HashSet<Id> {
    let dist_thresh = self.params.distance_threshold;
    let degree_bound = self.params.degree_bound;

    self.inst(|| VamanaInstrumentationEvent::RobustPruneBegin {
      node: node_id,
      candidates: candidate_ids.iter().copied().collect(),
    });

    let mut candidates = candidate_ids
      .into_par_iter()
      .map(|candidate_id| PointDist {
        id: candidate_id,
        dist: self.dist(node_id, candidate_id),
      })
      .collect::<VecDeque<_>>();
    candidates
      .make_contiguous()
      .sort_unstable_by_key(|s| OrderedFloat(s.dist));

    let mut new_neighbors = HashSet::new();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some(PointDist { id: p_star, .. }) = candidates.pop_front() {
      assert!(new_neighbors.insert(p_star));
      if new_neighbors.len() == degree_bound {
        self.inst(|| VamanaInstrumentationEvent::RobustPruneIteration {
          node: node_id,
          p_star,
          final_candidates: None,
        });
        break;
      }
      candidates.retain(|s| {
        let s_to_p = s.dist;
        let s_to_p_star = self.dist(p_star, s.id);
        s_to_p <= s_to_p_star * dist_thresh
      });
      self.inst(|| VamanaInstrumentationEvent::RobustPruneIteration {
        node: node_id,
        p_star,
        final_candidates: Some(candidates.iter().map(|s| s.id).collect()),
      });
    }
    new_neighbors
  }

  // The point referenced by each ID should already be inserted into the DB.
  // This is used when inserting, but also during initialization, so this is a separate function from `insert`.
  // WARNING: The graph must not be mutated while this function is executing, but it is up to the caller to ensure this.
  /// WARNING: This is publicly exposed, but use this only if you know what you're doing. There is no guarantees of API stability.
  pub fn optimize(&self, mut ids: Vec<Id>, mut metrics: Option<&mut OptimizeMetrics>) {
    // Shuffle to reduce chance of inserting around the same area in latent space.
    ids.shuffle(&mut thread_rng());
    for batch in ids.chunks(self.params.update_batch_size) {
      #[derive(Default)]
      struct Update {
        // These two aren't the same and can't be merged, as otherwise we can't tell whether we are supposed to replace or merge with the existing out-neighbors.
        replacement_base: Option<HashSet<Id>>,
        additional_edges: HashSet<Id>,
      }
      let updates = DashMap::<Id, Update>::new();

      self.inst(|| VamanaInstrumentationEvent::OptimizeBatchBegin {
        batch_ids: batch.to_vec(),
      });

      batch.into_par_iter().for_each(|&id| {
        // TODO Delete if already exists.

        // Initial GreedySearch.
        let mut candidates = self
          .greedy_search(
            IdOrPoint::Id(id),
            1,
            self.params.update_search_list_cap,
            |_| true,
            None,
            None,
          )
          .1;

        // RobustPrune.
        // RobustPrune requires locking the graph node at this point; we're already holding the lock so we're good to go.
        for n in self.ds.get_out_neighbors(id).unwrap_or_default() {
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
          .unwrap_or_else(|| self.ds.get_out_neighbors(id).unwrap());
        for j in u.additional_edges {
          new_neighbors.insert(j);
        }
        if new_neighbors.len() > self.params.degree_bound {
          new_neighbors = self.compute_robust_pruned(id, new_neighbors);
        };
        self.ds.set_out_neighbors(id, new_neighbors);
      });
    }
  }

  pub fn insert(&self, points: Vec<(Id, Array1<T>)>) {
    let ids = points.iter().map(|(id, _)| *id).collect_vec();
    points
      .into_par_iter()
      .for_each(|(id, point)| self.ds.set_point(id, point));
    self.optimize(ids, None);
  }

  pub fn query_with_filter(
    &self,
    query: &ArrayView1<T>,
    k: usize,
    filter: impl Fn(PointDist) -> bool,
  ) -> Vec<PointDist> {
    self
      .greedy_search(
        IdOrPoint::Point(query),
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
        IdOrPoint::Point(query),
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
  use crate::common::metric_euclidean;
  use crate::vamana::InMemoryVamana;
  use ahash::HashSet;
  use itertools::Itertools;
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
        update_batch_size: 64,
        update_search_list_cap: search_list_cap,
      },
      10_000,
      None,
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
