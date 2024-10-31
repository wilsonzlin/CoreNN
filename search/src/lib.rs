use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashSet;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BinaryHeap;
use strum_macros::Display;
use strum_macros::EnumString;

pub type Id = usize;
pub type Metric<T> = fn(&ArrayView1<T>, &ArrayView1<T>) -> f64;

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display, EnumString)]
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
  fn get_out_neighbors(&self, id: Id) -> Vec<Id>;
}

// Optimised custom function for k=1 and search_list_cap=1.
pub fn greedy_search_fast1<T: Scalar + Send + Sync>(
  graph: &impl GreedySearchable<T>,
  query: &ArrayView1<T>,
  metric: Metric<T>,
  start: Id,
  filter: impl (Fn(Id) -> bool) + Send + Sync,
) -> Option<Id> {
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
    graph
      .get_out_neighbors(cur)
      .into_par_iter()
      .filter(|n| !seen.contains(n))
      .map(|n| PointDist {
        id: n,
        dist: metric(&graph.get_point(cur).view(), &graph.get_point(n).view()),
      })
      .for_each(|n| {
        // If this node is a neighbor of a future expanded node, we don't need to compare the distance to this node, as if it's not the shortest now, it won't be then either.
        seen.insert(n.id);
        let mut s = state.lock();
        if filter(n.id) && !s.optima.is_some_and(|o| n.dist >= o.dist) {
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
  state.into_inner().optima.map(|n| n.id)
}

pub fn find_shortest_spanning_path<T: Scalar + Send + Sync>(
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

    // We could just search the other shards here, instead of storing the path. However, we do it this way to emulate how it works in reality, as it's unlikely we can load all shards into memory at once.
    path.push((from, to));

    // Move on to neighbors of `to` in the base shard.
    let new = graph
      .get_out_neighbors(to)
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
