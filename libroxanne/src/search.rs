use crate::common::Dtype;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use crate::common::PrecomputedDists;
use ahash::HashSet;
use ahash::HashSetExt;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Borrow;
use std::cmp::max;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::convert::identity;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
pub enum Query<'a, 'b, T: Dtype> {
  Id(Id),
  Vec(&'a ArrayView1<'b, T>),
}

impl<'a, 'b, T: Dtype> From<Id> for Query<'a, 'b, T> {
  fn from(id: Id) -> Self {
    Self::Id(id)
  }
}

impl<'a, 'b, T: Dtype> From<&'a ArrayView1<'b, T>> for Query<'a, 'b, T> {
  fn from(value: &'a ArrayView1<'b, T>) -> Self {
    Self::Vec(value)
  }
}

pub struct QueryBuilder<'a, T: Dtype, G: GreedySearchable<'a, T>> {
  g: &'a G,
  query: Query<'a, 'a, T>,
  k: usize,
  search_list_cap: usize,
  beam_width: usize,
  start: Id,
  filter: Option<Box<dyn Fn(PointDist) -> bool + 'a>>,
  out_visited: Option<&'a mut HashSet<Id>>,
  out_metrics: Option<&'a mut SearchMetrics>,
  ground_truth: Option<&'a HashSet<Id>>,
}

impl<'a, T: Dtype, G: GreedySearchable<'a, T>> QueryBuilder<'a, T, G> {
  pub fn beam_width(mut self, beam_width: usize) -> Self {
    self.beam_width = beam_width;
    self
  }

  pub fn filter(mut self, filter: impl Fn(PointDist) -> bool + 'a) -> Self {
    self.filter = Some(Box::new(filter));
    self
  }

  pub fn ground_truth(mut self, ground_truth: &'a HashSet<Id>) -> Self {
    self.ground_truth = Some(ground_truth);
    self
  }

  pub fn out_metrics(mut self, out_metrics: &'a mut SearchMetrics) -> Self {
    self.out_metrics = Some(out_metrics);
    self
  }

  pub fn out_visited(mut self, out_visited: &'a mut HashSet<Id>) -> Self {
    self.out_visited = Some(out_visited);
    self
  }

  pub fn search_list_cap(mut self, search_list_cap: usize) -> Self {
    self.search_list_cap = search_list_cap;
    self
  }

  pub fn start(mut self, start: Id) -> Self {
    self.start = start;
    self
  }

  pub fn query(self) -> Vec<PointDist> {
    self.g.greedy_search(
      self.query,
      self.k,
      self.search_list_cap,
      self.beam_width,
      self.start,
      |n| self.filter.as_ref().is_none_or(|f| f(n)),
      self.out_visited,
      self.out_metrics,
      self.ground_truth,
    )
  }
}

pub fn insert_into_ordered_vecdeque<T: Clone, K: Ord>(
  dest: &mut VecDeque<T>,
  src: &[T],
  key: impl Fn(&T) -> K,
) {
  let mut positions = Vec::new();
  for (i, v) in src.into_iter().enumerate() {
    let pos = dest
      .binary_search_by(|s| key(s).cmp(&key(v)))
      .map_or_else(identity, identity);
    positions.push((pos, i));
  }
  positions.sort_unstable();
  for (dest_i, src_i) in positions.into_iter().rev() {
    dest.insert(dest_i, src[src_i].clone());
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

// These generics allow for owned and borrowed variants in various forms (e.g. DashMap::Ref, slice, ArrayView, &Array); the complexity allows for avoiding copying where possible, which gets really expensive for in-memory usages, while supporting copied owned data for reading from disk.
pub trait GreedySearchable<'a, T: Dtype>: Sized {
  type Point: Borrow<Array1<T>>;
  type Neighbors: Borrow<Vec<Id>>;
  type FullVec: Borrow<Array1<T>>;

  fn medoid(&self) -> Id;
  fn metric(&self) -> Metric<T>;
  fn get_point(&'a self, id: Id) -> Self::Point;
  // If the graph supports it, provide full embedding for reranking as second return value.
  fn get_out_neighbors(&'a self, id: Id) -> (Self::Neighbors, Option<Self::FullVec>);
  fn precomputed_dists(&self) -> Option<&PrecomputedDists>;
  fn default_query_search_list_cap(&self) -> usize {
    150
  }
  fn default_beam_width(&self) -> usize {
    1
  }

  /// Calculate the distance between two points.
  fn dist(&'a self, a: Id, b: Id) -> f64 {
    self
      .precomputed_dists()
      .map(|pd| pd.get(a, b))
      .unwrap_or_else(|| {
        self.metric()(
          &self.get_point(a).borrow().view(),
          &self.get_point(b).borrow().view(),
        )
      })
  }

  /// Calculate the distance between a point and a query
  fn dist2(&'a self, a: Id, b: Query<T>) -> f64 {
    match b {
      Query::Id(b) => self.dist(a, b),
      Query::Vec(b) => self.metric()(&self.get_point(a).borrow().view(), b),
    }
  }

  /// Calculate the distance between a vector and a query.
  fn dist3(&'a self, a: &ArrayView1<T>, b: Query<T>) -> f64 {
    match b {
      Query::Id(b) => self.metric()(a, &self.get_point(b).borrow().view()),
      Query::Vec(b) => self.metric()(a, b),
    }
  }

  // Optimised custom function for k=1 and search_list_cap=1.
  fn greedy_search_fast1(
    &'a self,
    query: Query<T>,
    start: Id,
    filter: impl (Fn(Id) -> bool),
  ) -> Option<PointDist> {
    // We traverse filtered nodes but don't allow them as the answer. This is better than ignoring them while traversing, as that breaks the graph (poor/no navigability). However, it's not as exhaustive as a typical search as we do not backtrack if we have found a local optima but it's filtered, as that may cause excessive backtracking and a regression to a full graph scan (e.g. consider a scenario where the only node left is one that is actually the furtherest from the query). For our usages, it's fine to be "approximate" and even if not the most accurate node is returned (or None is returned), as overall it still works out.
    // TODO Study impact of performance and accuracy compared to full exhaustive search with backtracking while filtered.
    let mut cur = PointDist {
      id: start,
      dist: self.dist2(start, query),
    };
    // The optima cannot default to `start` as it may be filtered.
    let mut optima: Option<PointDist> = None;
    let mut seen = HashSet::<Id>::new();
    seen.insert(start);
    loop {
      let cur_id = cur.id;
      let (neighbors, full_vec) = self.get_out_neighbors(cur_id);
      if let Some(v) = full_vec {
        cur.dist = self.dist3(&v.borrow().view(), query);
      };
      for &n_id in neighbors.borrow() {
        // If this node is a neighbor of a future expanded node, we don't need to compare the distance to this node, as if it's not the shortest now, it won't be then either.
        if !seen.insert(n_id) {
          continue;
        };
        let n = PointDist {
          id: n_id,
          dist: self.dist2(n_id, query),
        };
        // Call filter before lock to reduce contention.
        let not_filtered = filter(n.id);
        if not_filtered && !optima.is_some_and(|o| o.dist <= n.dist) {
          optima = Some(n);
        }
        if n.dist < cur.dist {
          cur = n;
        }
      }
      if cur.id == cur_id {
        // No change, reached local optima.
        break;
      }
    }
    optima
  }

  // DiskANN paper, Algorithm 1: GreedySearch.
  // Returns a pair: (closest points, visited node IDs).
  // Filtered nodes will be visited and expanded but not considered for the final set of neighbors.
  fn greedy_search(
    &'a self,
    query: Query<T>,
    k: usize,
    search_list_cap: usize,
    beam_width: usize,
    start: Id,
    filter: impl Fn(PointDist) -> bool,
    mut out_visited: Option<&mut HashSet<Id>>,
    // (map from ID to row/col no, matrix of dists flattened).
    mut out_metrics: Option<&mut SearchMetrics>,
    ground_truth: Option<&HashSet<Id>>,
  ) -> Vec<PointDist> {
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
      let mut new_visited = (0..beam_width)
        .filter_map(|_| l_unvisited.pop_front())
        .collect::<Vec<_>>();
      let mut neighbors = HashSet::new();
      new_visited.iter_mut().for_each(|p_star| {
        let (p_neighbors, full_vec) = self.get_out_neighbors(p_star.id);
        if let Some(v) = full_vec {
          p_star.dist = self.dist3(&v.borrow().view(), query);
        }
        for &j in p_neighbors.borrow() {
          neighbors.insert(j);
        }
      });
      // Move to visited section.
      all_visited.extend(new_visited.iter().map(|e| e.id));
      insert_into_ordered_vecdeque(
        &mut l_visited,
        &new_visited
          .iter()
          .filter(|e| filter(**e))
          .cloned()
          .collect_vec(),
        |s| OrderedFloat(s.dist),
      );

      let new_unvisited = neighbors
        .iter()
        .filter_map(|&neighbor| {
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
        .collect_vec();
      l_unvisited_set.extend(new_unvisited.iter().map(|e| e.id));
      insert_into_ordered_vecdeque(&mut l_unvisited, &new_unvisited, |s| OrderedFloat(s.dist));

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

  fn find_shortest_spanning_tree(&'a self, start: Id) -> Vec<(Id, Id)> {
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
      let new = self
        .get_out_neighbors(to)
        .0
        .borrow()
        .iter()
        .filter_map(|&neighbor| {
          if visited.contains(&neighbor) {
            return None;
          }
          let dist = self.dist(to, neighbor);
          // Use negative dist as BinaryHeap is a max-heap.
          Some((OrderedFloat(-dist), to, neighbor))
        })
        .collect::<Vec<_>>();
      pq.extend(new);
    }
    path
  }

  fn query_builder(
    &'a self,
    query: impl Into<Query<'a, 'a, T>>,
    k: usize,
  ) -> QueryBuilder<'a, T, Self> {
    QueryBuilder {
      g: self,
      query: query.into(),
      k,
      search_list_cap: max(self.default_query_search_list_cap(), k),
      beam_width: self.default_beam_width(),
      start: self.medoid(),
      filter: None,
      out_visited: None,
      out_metrics: None,
      ground_truth: None,
    }
  }
}
