#![feature(f16)]

use ahash::HashMap;
use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashSet;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use strum_macros::Display;
use strum_macros::EnumString;

pub type Id = usize;
pub type Metric<T> = fn(&ArrayView1<T>, &ArrayView1<T>) -> f64;

#[derive(Clone, Copy)]
pub enum IdOrVec<'a, 'b, T: Scalar> {
  Id(Id),
  Vec(&'a ArrayView1<'b, T>),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PointDist {
  pub id: Id,
  pub dist: f64,
}

// A metric implementation of the Euclidean distance.
pub fn metric_euclidean<T: Scalar>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let diff = a - b;
  let squared_diff = &diff * &diff;
  let sum_squared_diff = squared_diff.sum();
  sum_squared_diff.to_f64().unwrap().sqrt()
}

// A metric implementation of the cosine distance (NOT similarity).
pub fn metric_cosine<T: Scalar>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let dot_product = a.dot(b).to_f64().unwrap();

  let a_norm = a.dot(a).to_f64().unwrap();
  let b_norm = b.dot(b).to_f64().unwrap();

  let denominator = (a_norm * b_norm).sqrt();

  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product / denominator
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display, EnumString, Serialize, Deserialize)]
pub enum StdMetric {
  L2,
  Cosine,
}

impl StdMetric {
  pub fn get_fn<T: Scalar>(self) -> fn(&ArrayView1<T>, &ArrayView1<T>) -> f64 {
    match self {
      StdMetric::L2 => metric_euclidean::<T>,
      StdMetric::Cosine => metric_cosine::<T>,
    }
  }
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

pub trait GreedySearchable<T: Scalar + Send + Sync>: Send + Sync {
  fn get_point(&self, id: Id) -> Array1<T>;
  // If the graph supports it, provide full embedding for reranking as second return value.
  fn get_out_neighbors(&self, id: Id) -> (Vec<Id>, Option<Array1<T>>);
}

// Optimised custom function for k=1 and search_list_cap=1.
pub fn greedy_search_fast1<T: Scalar + Send + Sync>(
  graph: &impl GreedySearchable<T>,
  query: &ArrayView1<T>,
  metric: Metric<T>,
  start: Id,
  filter: impl (Fn(Id) -> bool) + Send + Sync,
) -> Option<PointDist> {
  struct State {
    cur: PointDist,
    optima: Option<PointDist>,
  }
  // We traverse filtered nodes but don't allow them as the answer. This is better than ignoring them while traversing, as that breaks the graph (poor/no navigability). However, it's not as exhaustive as a typical search as we do not backtrack if we have found a local optima but it's filtered, as that may cause excessive backtracking and a regression to a full graph scan (e.g. consider a scenario where the only node left is one that is actually the furtherest from the query). For our usages, it's fine to be "approximate" and even if not the most accurate node is returned (or None is returned), as overall it still works out.
  // TODO Study impact of performance and accuracy compared to full exhaustive search with backtracking while filtered.
  let state = Mutex::new(State {
    cur: PointDist {
      id: start,
      dist: metric(&graph.get_point(start).view(), query),
    },
    // The optima cannot default to `start` as it may be filtered.
    optima: None,
  });
  let seen = DashSet::<Id>::new();
  seen.insert(start);
  loop {
    let cur = state.lock().cur.id;
    let (neighbors, full_vec) = graph.get_out_neighbors(cur);
    if let Some(v) = full_vec {
      state.lock().cur.dist = metric(&v.view(), query);
    };
    neighbors
      .into_par_iter()
      .filter(|n| !seen.contains(n))
      .map(|n| PointDist {
        id: n,
        dist: metric(&graph.get_point(cur).view(), &graph.get_point(n).view()),
      })
      .for_each(|n| {
        // If this node is a neighbor of a future expanded node, we don't need to compare the distance to this node, as if it's not the shortest now, it won't be then either.
        seen.insert(n.id);
        // Call filter before lock to reduce contention.
        let not_filtered = filter(n.id);
        let mut s = state.lock();
        if not_filtered && !s.optima.is_some_and(|o| o.dist <= n.dist) {
          s.optima = Some(n);
        }
        if n.dist < s.cur.dist {
          s.cur = n;
        }
      });
    if state.lock().cur.id == cur {
      // No change, reached local optima.
      break;
    }
  }
  state.into_inner().optima
}

// DiskANN paper, Algorithm 1: GreedySearch.
// Returns a pair: (closest points, visited node IDs).
// Filtered nodes will be visited and expanded but not considered for the final set of neighbors.
pub fn greedy_search<T: Scalar + Send + Sync>(
  graph: &impl GreedySearchable<T>,
  query: IdOrVec<T>,
  k: usize,
  search_list_cap: usize,
  beam_width: usize,
  metric: Metric<T>,
  start: Id,
  filter: impl Fn(PointDist) -> bool,
  mut out_visited: Option<&mut HashSet<Id>>,
  // (map from ID to row/col no, matrix of dists flattened).
  precomputed_dists: Option<Arc<(HashMap<Id, usize>, Vec<f16>)>>,
  mut out_metrics: Option<&mut SearchMetrics>,
  ground_truth: Option<&HashSet<Id>>,
) -> Vec<PointDist> {
  assert!(
    search_list_cap >= k,
    "search list capacity must be greater than or equal to k"
  );
  let calc_dist = |a: Id| match query {
    IdOrVec::Id(b) => match precomputed_dists.as_ref() {
      None => metric(&graph.get_point(a).view(), &graph.get_point(b).view()),
      Some(pd) => {
        let (id_to_no, dists) = pd.as_ref();
        let n = id_to_no.len();
        let ia = id_to_no[&a];
        let ib = id_to_no[&b];
        dists[ia * n + ib] as f64
      }
    },
    IdOrVec::Vec(v) => metric(v, &graph.get_point(a).view()),
  };

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
    dist: calc_dist(start),
  });
  let ground_truth_found = AtomicUsize::new(0);
  while !l_unvisited.is_empty() {
    let mut new_visited = (0..beam_width)
      .filter_map(|_| l_unvisited.pop_front())
      .collect::<Vec<_>>();
    let neighbors = DashSet::new();
    new_visited.par_iter_mut().for_each(|p_star| {
      let (p_neighbors, full_vec) = graph.get_out_neighbors(p_star.id);
      if let Some(v) = full_vec {
        p_star.dist = match query {
          IdOrVec::Id(id) => metric(&v.view(), &graph.get_point(id).view()),
          IdOrVec::Vec(o) => metric(&v.view(), o),
        };
      }
      for j in p_neighbors {
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
          dist: calc_dist(neighbor),
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

    if let Some(m) = &mut out_metrics {
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
  if let Some(out) = &mut out_visited {
    out.extend(all_visited);
  };
  closest
}

pub fn find_shortest_spanning_tree<T: Scalar + Send + Sync>(
  graph: &impl GreedySearchable<T>,
  metric: Metric<T>,
  start: Id,
) -> Vec<(Id, Id)> {
  let mut visited = HashSet::<Id>::new();
  let mut path = Vec::<(Id, Id)>::new();
  // Why use Dijkstra instead of simply calculating min(dist_of_in_neighbors_edges) for each node? After all, we traverse and expand every node in both, but the latter can be a bit more parallel. The reason is that the latter will create lots of symmetric edges, because if A is closest to B then so is B to A, but this is bad for forming a dependency path to iterate through the graph as we'll frequently stop due to A already being visited once we get to B. Basically, we use Dijkstra because it has a `visited` set that forces a non-cyclic continuous path.
  let mut pq = BinaryHeap::<(OrderedFloat<f64>, Id, Id)>::new();
  pq.push((OrderedFloat(0.0), start, start));
  while let Some((_, from, to)) = pq.pop() {
    if !visited.insert(to) {
      // We've already visited (there was a shorter path to this node).
      continue;
    };

    path.push((from, to));

    // Move on to neighbors of `to` in the base shard.
    let new = graph
      .get_out_neighbors(to)
      .0
      .into_par_iter()
      .filter_map(|neighbor| {
        if visited.contains(&neighbor) {
          return None;
        }
        let dist = metric(
          &graph.get_point(to).view(),
          &graph.get_point(neighbor).view(),
        );
        // Use negative dist as BinaryHeap is a max-heap.
        Some((OrderedFloat(-dist), to, neighbor))
      })
      .collect::<Vec<_>>();
    pq.extend(new);
  }
  path
}
