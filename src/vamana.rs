use ahash::AHashMap;
use ahash::AHashSet;
use croaring::Bitmap;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::VecDeque;
use std::iter::zip;

// More than 4B is unlikely to scale well on a single shard. Using u32 instead of u64 allows us to use fast RoaringBitmaps everywhere.
type Id = u32;
type Metric<P> = fn(&P, &P) -> f64;

struct PointDist {
  id: Id,
  dist: f64,
}

// A metric implementation of the Euclidean distance.
fn metric_euclidean<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
  zip(a, b).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
}

// A metric implementation of the cosine similarity.
fn metric_cosine<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
  let dot = zip(a, b).map(|(a, b)| a * b).sum::<f64>();
  let norm_a = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
  let norm_b = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
  1.0 - dot / (norm_a * norm_b)
}

// DiskANN paper, Algorithm 1: GreedySearch.
fn greedy_search<P>(
  adj_list: &AHashMap<Id, Bitmap>,
  id_to_point: &AHashMap<Id, P>,
  metric: Metric<P>,
  // This corresponds to `s` in the paper.
  start: Id,
  // This corresponds to `x_q` in the paper.
  query: &P,
  k: usize,
  // Must be greater than `k` (according to paper). This corresponds to `L` in the paper.
  beam_width: usize,
) -> (Vec<PointDist>, Bitmap) {
  assert!(beam_width > k, "beam width must be greater than k");

  // It's too inefficient to calculate argmin(dist(l\V, query)) repeatedly.
  // Instead, we'll use a sorted array for O(1) argmin.
  // We don't use a BinaryHeap as a priority queue does not allow for efficient truncation, which we do at each iteration. A Vec may be slower in theory to insert, but `k` is often small enough that it's actually faster in reality due to cache locality.
  // This will also ensure there are no duplicates. (Important, as the algorithm in the DiskANN paper specifies that this is a set.)
  struct BeamQueue {
    queue: VecDeque<PointDist>,
    set: Bitmap,
    k: usize,
  }

  impl BeamQueue {
    pub fn new(k: usize) -> Self {
      Self {
        queue: VecDeque::new(),
        set: Bitmap::new(),
        k,
      }
    }

    pub fn push(&mut self, state: PointDist) {
      debug_assert!(self.queue.len() <= self.k);
      debug_assert!(self.set.cardinality() as usize <= self.k);
      // Do not insert if already exists.
      if self.set.contains(state.id) {
        return;
      }
      let pos = match self
        .queue
        .binary_search_by(|s| s.dist.partial_cmp(&state.dist).unwrap())
      {
        Ok(pos) => pos,
        Err(pos) => pos,
      };
      // Don't bother inserting if it'll be pruned anyway.
      if pos >= self.k {
        return;
      }
      self.set.add(state.id);
      self.queue.insert(pos, state);
      if self.queue.len() > self.k {
        let PointDist { id, .. } = self.queue.pop_back().unwrap();
        self.set.remove(id);
      }
    }

    pub fn pop(&mut self) -> Option<PointDist> {
      let s = self.queue.pop_front();
      if let Some(s) = s.as_ref() {
        self.set.remove(s.id);
      }
      s
    }
  }

  // It's too inefficient to calculate L\V repeatedly.
  // Since we need both L (return value) and L\V (each iteration), we split L into V and ¬V.
  // For simplicity, we'll just allow both to reach `k` size, and do a final merge at the end. This doubles the memory requirements, but in reality `k` is often small enough that it's not a problem.
  // We need both to reach up to `k` as in the worst case all `k` are in exactly one of them.
  // We also need `all_visited` as `l_visited` truncates to `k`, but we also want all visited points in the end.
  // L = l_visited + l_unvisited
  let mut l_unvisited = BeamQueue::new(beam_width); // L \ V
  let mut l_visited = BeamQueue::new(beam_width); // V
  let mut all_visited = Bitmap::new();
  l_unvisited.push(PointDist {
    id: start,
    dist: metric(&id_to_point[&start], query),
  });
  while let Some(PointDist { id, dist }) = l_unvisited.pop() {
    // Move to visited section.
    l_visited.push(PointDist { id, dist });
    all_visited.add(id);
    for neighbor in adj_list[&id].iter() {
      // We separate L out into V and not V, so we must manually ensure the property that l_visited and l_unvisited are disjoint.
      if !all_visited.contains(neighbor) {
        let dist_to_query = metric(&id_to_point[&neighbor], query);
        l_unvisited.push(PointDist { id: neighbor, dist });
      }
    }
  }

  // Find the k closest points from both l_visited + l_unvisited (= L).
  let mut closest = Vec::new();
  while closest.len() < k {
    match (l_visited.pop(), l_unvisited.pop()) {
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
  (closest, all_visited)
}

// DiskANN paper, Algorithm 2: RobustPrune.
fn robust_prune<P>(
  adj_list: &mut AHashMap<Id, Bitmap>,
  id_to_point: &AHashMap<Id, P>,
  metric: Metric<P>,
  point: Id,
  // This corresponds to `V` in the paper.
  candidates: Bitmap,
  // Must be >= 1. This corresponds to `α` in the paper.
  distance_threshold: f64,
  // This corresponds to `R` in the paper.
  degree_bound: usize,
) {
  assert!(
    distance_threshold >= 1.0,
    "distance threshold must be at least 1"
  );

  let mut candidates = candidates
    .or(&adj_list[&point])
    .iter()
    .filter(|&id| id != point)
    .map(|id| PointDist {
      id,
      dist: metric(&id_to_point[&id], &id_to_point[&point]),
    })
    .sorted_unstable_by_key(|s| OrderedFloat(s.dist))
    .collect::<VecDeque<_>>();

  let point_neighbors = adj_list.get_mut(&point).unwrap();
  point_neighbors.clear();
  // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
  while let Some(PointDist { id: p_star, .. }) = candidates.pop_front() {
    point_neighbors.add(p_star);
    if point_neighbors.cardinality() as usize == degree_bound {
      break;
    }
    candidates.retain(|s| {
      let dist_to_p_star = metric(&id_to_point[&s.id], &id_to_point[&p_star]);
      distance_threshold * dist_to_p_star > s.dist
    });
  }
}

#[derive(Clone)]
pub struct VamanaParams {
  // The `R` parameter in the DiskANN paper, section 2.3: the initial number of randomly chosen out-neighbors for each point. The paper recommends at least log(N), where N is the number of points.
  init_neighbors: usize,
  // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
  medoid_sample_size: usize,
  beam_width: usize,
  // Corresponds to `α` in the DiskANN paper. Must be at least 1.
  distance_threshold: f64,
  // Corresponds to `R` in the DiskANN paper.
  degree_bound: usize,
}

pub struct Vamana<P: Clone> {
  adj_list: AHashMap<Id, Bitmap>,
  id_to_point: AHashMap<Id, P>,
  metric: fn(&P, &P) -> f64,
  medoid: Id,
  params: VamanaParams,
}

impl<P: Clone> Vamana<P> {
  pub fn new(ids: &[Id], points: &[P], metric: Metric<P>, params: VamanaParams) -> Self {
    assert_eq!(
      ids.len(),
      points.len(),
      "ID count and point count do not equal"
    );
    let id_to_point = zip(ids, points)
      .map(|(id, p)| (*id, p.clone()))
      .collect::<AHashMap<_, _>>();

    let adj_list = ids
      .iter()
      .map(|&id| {
        let mut rng = thread_rng();
        let neighbors = ids
          .choose_multiple(&mut rng, params.init_neighbors)
          .copied()
          .collect();
        (id, neighbors)
      })
      .collect();

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = ids
      .choose_multiple(&mut thread_rng(), params.medoid_sample_size)
      .copied()
      .min_by_key(|id| {
        OrderedFloat(
          points
            .iter()
            .map(|p| metric(&id_to_point[&id], p))
            .sum::<f64>(),
        )
      })
      .unwrap();

    Self {
      adj_list,
      id_to_point,
      medoid,
      metric,
      params,
    }
  }

  pub fn index(&mut self) {
    let params = &self.params;

    // Iterate points in random order.
    let mut ids_rand = self.id_to_point.keys().copied().collect_vec();
    ids_rand.shuffle(&mut thread_rng());
    for id in ids_rand {
      let (_closest, visited) = greedy_search(
        &self.adj_list,
        &self.id_to_point,
        self.metric,
        self.medoid,
        &self.id_to_point[&id],
        1,
        params.beam_width,
      );
      robust_prune(
        &mut self.adj_list,
        &self.id_to_point,
        self.metric,
        id,
        visited,
        params.distance_threshold,
        params.degree_bound,
      );
      // We will update the graph, so to satisfy Rust, we must clone now. (As a human, I can say that `j` should never equal `id` so this would be safe without cloning, but the compiler doesn't know.)
      for j in self.adj_list[&id].clone().iter() {
        // It's safe to always add `id` and save a clone + add, even though this differs from Algorithm 3 in the paper.
        // - In the first branch, `robust_prune` will immediately clear the neighbors.
        // - In the second branch, we will always add `id`.
        let j_neighbors = self.adj_list.get_mut(&j).unwrap();
        j_neighbors.add(id);
        if j_neighbors.cardinality() as usize > params.degree_bound {
          // Clone now instead of as inline arg so that we can drop the mut borrow to adj_list now and therefore mut borrow it in the first arg to robust_prune.
          let candidates = j_neighbors.clone();
          robust_prune(
            &mut self.adj_list,
            &self.id_to_point,
            self.metric,
            j,
            candidates,
            params.distance_threshold,
            params.degree_bound,
          );
        }
      }
    }
  }

  pub fn query(&self, query: &P, k: usize) -> Vec<PointDist> {
    greedy_search(
      &self.adj_list,
      &self.id_to_point,
      self.metric,
      self.medoid,
      query,
      k,
      self.params.beam_width,
    )
    .0
  }
}
