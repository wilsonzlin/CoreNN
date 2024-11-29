use crate::common::Dtype;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use crate::common::PrecomputedDists;
use crate::util::AsyncConcurrentIteratorExt;
use crate::util::AsyncConcurrentStreamExt;
use crate::util::CollectionExt;
use ahash::HashSet;
use ahash::HashSetExt;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::hash::Hash;

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

fn no_filter(_id: Id) -> bool {
  true
}

pub struct GreedySearchParams<'a, T: Dtype, F> {
  pub(crate) query: Query<'a, 'a, T>,
  pub(crate) k: usize,
  pub(crate) search_list_cap: usize,
  pub(crate) beam_width: usize,
  pub(crate) start: Id,
  pub(crate) filter: F,
  pub(crate) out_visited: Option<&'a mut HashSet<Id>>,
  pub(crate) out_metrics: Option<&'a mut SearchMetrics>,
  pub(crate) ground_truth: Option<&'a HashSet<Id>>,
}

impl<'a, T: Dtype, F: Fn(Id) -> bool> GreedySearchParams<'a, T, F> {
  pub fn new(
    query: impl Into<Query<'a, 'a, T>>,
    k: usize,
    start: Id,
  ) -> GreedySearchParams<'a, T, fn(Id) -> bool> {
    GreedySearchParams {
      query: query.into(),
      k,
      search_list_cap: k,
      beam_width: 1,
      start,
      filter: no_filter,
      out_visited: None,
      out_metrics: None,
      ground_truth: None,
    }
  }

  pub fn beam_width(mut self, beam_width: usize) -> Self {
    self.beam_width = beam_width;
    self
  }

  pub fn filter(mut self, filter: F) -> Self {
    self.filter = filter;
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

  pub async fn query_async(self, g: &impl GreedySearchableAsync<T>) -> Vec<PointDist> {
    g.greedy_search_async(self).await
  }

  pub fn query_sync(self, g: &impl GreedySearchableSync<T>) -> Vec<PointDist> {
    g.greedy_search_sync(self)
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

// Memory copying is really expensive, and can use a lot of unnecessary CPU (check profiler). But the issue is that graphs can be stored in many different types, so supporting some concept of "zero-cost" use, across references (e.g. in-memory) and owned values (e.g. copied after loading from disk), is hard.
// Things I'd like to be able to use as points:
// - Array1, ArrayView1
// - CowArray1
// - Array2.row(), ArrayView2.row()
// - DashMap::Ref, DashMap::RefMut
// Things I'd like to be able to use as neighbors:
// - Vec<Id>, Box<Id>, Cow<[Id]>, [Id; N], &[Id], &mut [Id]
// - HashSet<Id>, HashMap<Id, _>.keys()
// - DashSet<Id>
// - croaring::Bitmap
// - (0..n)
// - DashMap::Ref, DashMap::RefMut
// - mapped from lock-held value
// - ...anything iterable
// Basically, it could be owned, borrowing something self owns, passing the ref to something self also borrows, within some container like a lock guard, something that can be DeRef'd or AsRef'd or Borrow'd into any of these, a consumable single-use dynamic iterator, a view/slice within some larger thing, etc. This requires decent flexibility with whatever type system approach we use.
// Constraints:
// - Rust does not allow implementing foreign traits on foreign types, only own traits on foreign types or foreign traits on own types.
// - I'd prefer not to use some enum, which just offloads type-based dispatch to runtime and offers no extensibility to downstream users.
// Possible approaches:
// - Use a standard trait like Borrow<ArrayView1<T>>.
//   - Downsides: most custom types don't support this, so I have to implement a lot of newtype wrappers (due to constraint).
// - Create a new trait.
//   - Allows implementing on most types without using a bunch of newtype wrappers. Downstream users will still need to wrap in a newtype, but we should impl for most common external/std library types.

pub trait IVec<T: Dtype>: Sized {
  fn view(&self) -> ArrayView1<T>;

  fn into_vec(self) -> Vec<T> {
    self.view().to_vec()
  }
}

impl<T: Dtype> IVec<T> for Array1<T> {
  fn view(&self) -> ArrayView1<T> {
    self.view()
  }

  fn into_vec(self) -> Vec<T> {
    // TODO The Array may have a different stride and offset.
    self.into_raw_vec()
  }
}

impl<T: Dtype> IVec<T> for &Array1<T> {
  fn view(&self) -> ArrayView1<T> {
    Array1::view(self)
  }
}

impl<T: Dtype> IVec<T> for Cow<'_, [T]> {
  fn view(&self) -> ArrayView1<T> {
    ArrayView1::from(&*self)
  }

  fn into_vec(self) -> Vec<T> {
    self.into_owned()
  }
}

impl<T: Dtype> IVec<T> for &'_ [T] {
  fn view(&self) -> ArrayView1<T> {
    ArrayView1::from(self)
  }

  fn into_vec(self) -> Vec<T> {
    self.to_vec()
  }
}

impl<T: Dtype> IVec<T> for Vec<T> {
  fn view(&self) -> ArrayView1<T> {
    ArrayView1::from(self.as_slice())
  }

  fn into_vec(self) -> Vec<T> {
    self
  }
}

impl<'r, K: Eq + Hash, T: Dtype> IVec<T> for dashmap::mapref::one::Ref<'r, K, Array1<T>> {
  fn view(&self) -> ArrayView1<T> {
    self.value().view()
  }
}

// Why another trait instead of Iterator or IntoIterator?
// - Sometimes we need to support wrapped types like dashmap::Ref, which implements Deref<IntoIterator> but not IntoIterator, so it's either a new trait or a newtype implementing IntoIterator.
// - Some types iterate `&Id`, others `Id`; with our own trait, we can specify a return type of `impl IntoIterator<Item=Id>` for this trait's method so the implementer can use any complex chain of iterators if necessary. However, it's currently unstable to use `type Neighbors: impl IntoIterator<Item=Id>`.
// Unintended bonus: we can have self-moving trait methods for getting as owned Vec, HashSet, etc.; by default they just self.iter().collect(), but downstream implementers can use more optimized versions (e.g. the Self is already an owned Vec, so just move it, or is already a reference to one, so just clone it).
pub trait INeighbors: Sized {
  fn iter(&self) -> impl Iterator<Item = Id>;

  fn into_vec(self) -> Vec<Id> {
    self.iter().collect_vec()
  }
}

impl INeighbors for &[Id] {
  fn iter(&self) -> impl Iterator<Item = Id> {
    self.into_iter().map(|id| *id)
  }

  fn into_vec(self) -> Vec<Id> {
    self.to_vec()
  }
}

impl INeighbors for Vec<Id> {
  fn iter(&self) -> impl Iterator<Item = Id> {
    self.as_slice().iter().map(|id| *id)
  }

  fn into_vec(self) -> Vec<Id> {
    self
  }
}

impl INeighbors for Cow<'_, [Id]> {
  fn iter(&self) -> impl Iterator<Item = Id> {
    self.into_iter().map(|id| *id)
  }

  fn into_vec(self) -> Vec<Id> {
    self.into_owned()
  }
}

impl<'r, K: Eq + Hash> INeighbors for dashmap::mapref::one::Ref<'r, K, Vec<Id>> {
  fn iter(&self) -> impl Iterator<Item = Id> {
    self.value().iter()
  }

  fn into_vec(self) -> Vec<Id> {
    self.value().clone()
  }
}

#[derive(Default)]
pub struct GreedySearchState {
  // It's too inefficient to calculate L\V repeatedly.
  // Since we need both L (return value) and L\V (each iteration), we split L into V and ¬V.
  // For simplicity, we'll just allow both to reach `k` size, and do a final merge at the end. This doubles the memory requirements, but in reality `k` is often small enough that it's not a problem.
  // We also need `all_visited` as `l_visited` truncates to `k`, but we also want all visited points in the end.
  // L = l_visited + l_unvisited
  // `l_unvisited_set` is for members currently in l_unvisited, to avoid pushing duplicates. (NOTE: This is *not* the same as `all_visited`.) We don't need one for `l_visited` because we only push popped elements from `l_unvisited`, which we guarantee are unique (as previously mentioned).
  l_unvisited_set: HashSet<Id>,
  l_unvisited: VecDeque<PointDist>, // L \ V
  l_visited: VecDeque<PointDist>,   // V
  all_visited: HashSet<Id>,
  ground_truth_found: usize,
}

// Don't use `GreedySearchable<'a>` instead of `type XXX<'a>: ...` as the former constrains the entire trait, which causes lifetime issues within async contexts.
pub trait GreedySearchable<T: Dtype>: Send + Sync + Sized {
  // These generics allow for owned and borrowed variants in various forms (e.g. DashMap::Ref, slice, ArrayView, &Array); the complexity allows for avoiding copying where possible, which gets really expensive for in-memory usages, while supporting copied owned data for reading from disk.
  type Point<'a>: IVec<T>
  where
    Self: 'a;
  type Neighbors<'a>: INeighbors
  where
    Self: 'a;
  type FullVec: IVec<T>;

  fn medoid(&self) -> Id;
  fn metric(&self) -> Metric<T>;
  /// NOTE: This isn't I/O; all get_point uses memory only (even LTI, which uses cached PQ vecs). Therefore, don't use ThreadPool when calling this multiple times.
  fn get_point<'a>(&'a self, id: Id) -> Self::Point<'a>;
  fn precomputed_dists(&self) -> Option<&PrecomputedDists>;

  /// Calculate the distance between two points.
  /// NOTE: This isn't I/O. Therefore, don't use ThreadPool when calling this multiple times.
  fn dist(&self, a: Id, b: Id) -> f64 {
    if let Some(pd) = self.precomputed_dists().map(|pd| pd.get(a, b)) {
      return pd;
    };
    let a = self.get_point(a);
    let b = self.get_point(b);
    self.metric()(&a.view(), &b.view())
  }

  /// Calculate the distance between a point and a query.
  /// NOTE: This isn't I/O. Therefore, don't use ThreadPool when calling this multiple times.
  fn dist2(&self, a: Id, b: Query<T>) -> f64 {
    match b {
      Query::Id(b) => self.dist(a, b),
      Query::Vec(b) => self.metric()(&self.get_point(a).view(), b),
    }
  }

  /// Calculate the distance between a vector and a query.
  /// NOTE: This isn't I/O. Therefore, don't use ThreadPool when calling this multiple times.
  fn dist3(&self, a: &ArrayView1<T>, b: Query<T>) -> f64 {
    match b {
      Query::Id(b) => self.metric()(a, &self.get_point(b).view()),
      Query::Vec(b) => self.metric()(a, b),
    }
  }

  // We decompose the greedy search algorithm into subroutines that can be shared across both greedy_search_{a,}sync instead of duplicating a whole bunch of code across both variants. These decompositions are the gs_* methods (which should be only used internally).
  fn gs_init<F>(&self, params: &GreedySearchParams<T, F>) -> GreedySearchState {
    assert!(
      params.search_list_cap >= params.k,
      "search list capacity must be greater than or equal to k"
    );
    let mut s = GreedySearchState::default();
    s.l_unvisited.push_back(PointDist {
      id: params.start,
      dist: self.dist2(params.start, params.query),
    });
    s
  }

  /// If returns None, end loop.
  fn gs_loop_itertion_nodes_to_expand<F>(
    &self,
    state: &mut GreedySearchState,
    params: &GreedySearchParams<T, F>,
  ) -> Option<Vec<PointDist>> {
    let nodes = (0..params.beam_width)
      .filter_map(|_| state.l_unvisited.pop_front())
      .collect_vec();
    if nodes.is_empty() {
      None
    } else {
      Some(nodes)
    }
  }

  fn gs_loop_iteration_after_nodes_expanded<'a, F: Fn(Id) -> bool>(
    &'a self,
    expanded: impl IntoIterator<Item = (PointDist, Self::Neighbors<'a>, Option<Self::FullVec>)>,
    state: &mut GreedySearchState,
    params: &mut GreedySearchParams<T, F>,
  ) {
    let mut neighbors = HashSet::new();
    for (mut p_star, p_neighbors, full_vec) in expanded {
      if let Some(v) = full_vec {
        p_star.dist = self.dist3(&v.view(), params.query);
      }
      for j in p_neighbors.iter() {
        neighbors.insert(j);
      }
      // Move to visited section.
      state.all_visited.insert(p_star.id);
      if (params.filter)(p_star.id) {
        state
          .l_visited
          .insert_into_ordered(p_star, |s| OrderedFloat(s.dist));
      }
    }
    let neighbors_len = neighbors.len();

    let mut new_unvisited_len = 0;
    for neighbor in neighbors {
      // We separate L out into V and not V, so we must manually ensure the property that l_visited and l_unvisited are disjoint.
      if state.all_visited.contains(&neighbor) {
        continue;
      };
      if !state.l_unvisited_set.insert(neighbor) {
        continue;
      };
      if params.ground_truth.is_some_and(|gt| gt.contains(&neighbor)) {
        state.ground_truth_found += 1;
      }
      let dist = self.dist2(neighbor, params.query);
      state
        .l_unvisited
        .insert_into_ordered(PointDist { id: neighbor, dist }, |s| OrderedFloat(s.dist));
      new_unvisited_len += 1;
    }

    let mut dropped_unvisited = 0;
    let mut dropped_visited = 0;
    while state.l_unvisited.len() + state.l_visited.len() > params.search_list_cap {
      let (Some(u), Some(v)) = (state.l_unvisited.back(), state.l_visited.back()) else {
        break;
      };
      if u.dist >= v.dist {
        state.l_unvisited.pop_back();
        dropped_unvisited += 1;
      } else {
        state.l_visited.pop_back();
        dropped_visited += 1;
      }
    }

    if let Some(m) = &mut params.out_metrics {
      m.iterations.push(SearchIterationMetrics {
        dropped_candidates: neighbors_len - new_unvisited_len,
        dropped_unvisited,
        dropped_visited,
        ground_truth_found: state.ground_truth_found,
        new_candidates: new_unvisited_len,
        unvisited_dist_mins: state
          .l_unvisited
          .front()
          .map(|n| n.dist)
          .unwrap_or_default(),
        unvisited_dist_sum: state.l_unvisited.iter().map(|n| n.dist).sum(),
        visited: state.all_visited.len(),
      });
    }
  }

  fn gs_final<F>(
    &self,
    mut state: GreedySearchState,
    mut params: GreedySearchParams<T, F>,
  ) -> Vec<PointDist> {
    // Find the k closest points from both l_visited + l_unvisited (= L).
    let mut closest = Vec::new();
    while closest.len() < params.k {
      match (state.l_visited.pop_front(), state.l_unvisited.pop_front()) {
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
    closest.truncate(params.k);
    if let Some(out) = &mut params.out_visited {
      out.extend(state.all_visited);
    };
    closest
  }
}

pub trait GreedySearchableSync<T: Dtype>: GreedySearchable<T> {
  /// If the graph supports it, and get_point doesn't already provide the full embedding, provide full embedding for reranking as second return value.
  fn get_out_neighbors_sync<'a>(&'a self, id: Id) -> (Self::Neighbors<'a>, Option<Self::FullVec>);

  // DiskANN paper, Algorithm 1: GreedySearch.
  // Returns a pair: (closest points, visited node IDs).
  // Filtered nodes will be visited and expanded but not considered for the final set of neighbors.
  fn greedy_search_sync<F: Fn(Id) -> bool>(
    &self,
    mut params: GreedySearchParams<T, F>,
  ) -> Vec<PointDist> {
    let mut state = self.gs_init(&params);
    while let Some(nodes_to_expand) = self.gs_loop_itertion_nodes_to_expand(&mut state, &params) {
      let expanded = nodes_to_expand.into_iter().map(|p_star| {
        let (p_neighbors, full_vec) = self.get_out_neighbors_sync(p_star.id);
        (p_star, p_neighbors, full_vec)
      });
      self.gs_loop_iteration_after_nodes_expanded(expanded, &mut state, &mut params);
    }
    self.gs_final(state, params)
  }

  /// Optimised custom function for k=1 and search_list_cap=1 and beam_width=1.
  fn greedy_search_fast1(
    &self,
    query: Query<'_, '_, T>,
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
      let (neighbors, full_vec) = self.get_out_neighbors_sync(cur_id);
      if let Some(v) = full_vec {
        cur.dist = self.dist3(&v.view(), query);
      };
      for n_id in neighbors.iter() {
        // If this node is a neighbor of a future expanded node, we don't need to compare the distance to this node, as if it's not the shortest now, it won't be then either.
        if !seen.insert(n_id) {
          continue;
        };
        let n = PointDist {
          id: n_id,
          dist: self.dist2(n_id, query),
        };
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

  fn find_shortest_spanning_tree(&self, start: Id) -> Vec<(Id, Id)> {
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
        .get_out_neighbors_sync(to)
        .0
        .iter()
        .filter_map(|neighbor| {
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
}

/// See corresponding method comments on GreedySearchableSync.
pub trait GreedySearchableAsync<T: Dtype>: GreedySearchable<T> {
  async fn get_out_neighbors_async<'a>(
    &'a self,
    id: Id,
  ) -> (Self::Neighbors<'a>, Option<Self::FullVec>);

  async fn greedy_search_async<F: Fn(Id) -> bool>(
    &self,
    mut params: GreedySearchParams<'_, T, F>,
  ) -> Vec<PointDist> {
    let mut state = self.gs_init(&params);
    while let Some(nodes_to_expand) = self.gs_loop_itertion_nodes_to_expand(&mut state, &params) {
      let expanded = nodes_to_expand
        .into_iter()
        // Parallelize as the purpose of beam width > 1 is to read from disk in parallel.
        // Use map_concurrent and not tokio::spawn as we're mostly just awaiting I/O and don't need CPU parallelism.
        .map_concurrent_unordered(async |p_star| {
          let (p_neighbors, full_vec) = self.get_out_neighbors_async(p_star.id).await;
          (p_star, p_neighbors, full_vec)
        })
        .collect_vec()
        .await;
      self.gs_loop_iteration_after_nodes_expanded(expanded, &mut state, &mut params);
    }
    self.gs_final(state, params)
  }
}
