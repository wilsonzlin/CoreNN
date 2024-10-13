use crate::beamqueue::BeamQueue;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use arbitrary_lock::ArbitraryLock;
use croaring::Bitmap;
use dashmap::DashMap;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::collections::VecDeque;

// Return owned values:
// - We use DashMap for in-memory, so we can't return a ref while holding a lock in the map entry.
// - From disk, we copy the bytes.
pub trait VamanaDatastore<T: Scalar + Send + Sync>: Sync {
  fn get_point(&self, id: Id) -> Option<Array1<T>>;
  fn set_point(&self, id: Id, point: Array1<T>);
  fn get_out_neighbors(&self, id: Id) -> Option<Vec<Id>>;
  fn set_out_neighbors(&self, id: Id, neighbors: Vec<Id>);
}

#[derive(Default)]
pub struct InMemoryVamana<T: Scalar + Send + Sync> {
  adj_list: DashMap<Id, Vec<Id>>,
  id_to_point: DashMap<Id, Array1<T>>,
}

impl<T: Scalar + Send + Sync> VamanaDatastore<T> for InMemoryVamana<T> {
  fn get_point(&self, id: Id) -> Option<Array1<T>> {
    self.id_to_point.get(&id).map(|e| e.clone())
  }

  fn set_point(&self, id: Id, point: Array1<T>) {
    self.id_to_point.insert(id, point);
  }

  fn get_out_neighbors(&self, id: Id) -> Option<Vec<Id>> {
    self.adj_list.get(&id).map(|e| e.clone())
  }

  fn set_out_neighbors(&self, id: Id, neighbors: Vec<Id>) {
    self.adj_list.insert(id, neighbors);
  }
}

impl<T: Scalar + Send + Sync> InMemoryVamana<T> {
  pub fn init(
    dataset: Vec<(Id, Array1<T>)>,
    metric: Metric<T>,
    params: VamanaParams,
  ) -> Vamana<T, Self> {
    // Initialise to R-regular graph with random edges.
    #[cfg(feature = "verbose")]
    let progress = std::sync::atomic::AtomicUsize::new(0);
    let adj_list = DashMap::new();
    dataset.par_iter().for_each(|(id, _)| {
      #[cfg(feature = "verbose")]
      {
        let p = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if p % 100000 == 0 {
          println!("Build random graph: {}/{}", p, dataset.len());
        }
      }
      let mut rng = thread_rng();
      let neighbors = dataset
        .choose_multiple(&mut rng, params.degree_bound + 1) // Choose +1 in case we pick self.
        .map(|e| e.0)
        .filter(|oid| id != oid)
        .take(params.degree_bound)
        .collect_vec();
      adj_list.insert(*id, neighbors);
    });
    #[cfg(feature = "verbose")]
    println!("Build random graph: complete");

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = {
      let mut rng = thread_rng();
      let sample_nos = (0..params.medoid_sample_size)
        .map(|_| rng.gen_range(0..dataset.len()))
        .collect_vec();
      #[cfg(feature = "verbose")]
      println!("Calculate medoid: using {} points", sample_nos.len());
      let idx = sample_nos
        .par_iter()
        .copied()
        .min_by_key(|&i| {
          let p = dataset[i].1.view();
          OrderedFloat(
            sample_nos
              .iter()
              .map(|&j| metric(&p, &dataset[j].1.view()))
              .sum::<f64>(),
          )
        })
        .unwrap();
      dataset[idx].0
    };
    #[cfg(feature = "verbose")]
    println!("Calculate medoid: complete");

    // Iterate points in random order.
    // Build before `id_to_point` as that will take ownership.
    let mut insert_order = dataset.iter().map(|e| e.0).collect_vec();
    insert_order.shuffle(&mut thread_rng());

    let id_to_point = dataset.into_iter().collect();

    let graph = Vamana::new(
      Self {
        adj_list,
        id_to_point,
      },
      medoid,
      metric,
      params,
    );

    #[cfg(feature = "verbose")]
    let (progress, n) = {
      let n = insert_order.len();
      println!("Insert vectors: starting {} points", n);
      (std::sync::atomic::AtomicUsize::new(0), n)
    };
    insert_order.into_par_iter().for_each(|id| {
      graph._insert(id, None);
      #[cfg(feature = "verbose")]
      {
        let p = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if p % 10000 == 0 {
          println!("Insert vectors: {}/{}", p, n);
        }
      }
    });
    #[cfg(feature = "verbose")]
    println!("Insert vectors: complete");

    graph
  }
}

#[derive(Clone)]
pub struct VamanaParams {
  // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
  pub medoid_sample_size: usize,
  pub beam_width: usize,
  // Corresponds to `α` in the DiskANN paper. Must be at least 1.
  pub distance_threshold: f64,
  // Corresponds to `R` in the DiskANN paper. The paper recommends at least log(N), where N is the number of points.
  pub degree_bound: usize,
}

pub struct Vamana<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> {
  ds: DS,
  update_locker: ArbitraryLock<Id, parking_lot::Mutex<()>>,
  metric: Metric<T>,
  medoid: Id,
  params: VamanaParams,
}

impl<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> Vamana<T, DS> {
  pub fn new(ds: DS, medoid: Id, metric: Metric<T>, params: VamanaParams) -> Self {
    Self {
      ds,
      metric,
      medoid,
      params,
      update_locker: ArbitraryLock::new(),
    }
  }

  // DiskANN paper, Algorithm 1: GreedySearch.
  // Returns a pair: (closest points, visited node IDs).
  fn greedy_search(
    &self,
    // This corresponds to `s` in the paper.
    start: Id,
    // This corresponds to `x_q` in the paper.
    query: &ArrayView1<T>,
    k: usize,
    // Must be greater than `k` (according to paper). This corresponds to `L` in the paper.
    beam_width: usize,
  ) -> (Vec<PointDist>, Bitmap) {
    assert!(beam_width > k, "beam width must be greater than k");

    // It's too inefficient to calculate L\V repeatedly.
    // Since we need both L (return value) and L\V (each iteration), we split L into V and ¬V.
    // For simplicity, we'll just allow both to reach `k` size, and do a final merge at the end. This doubles the memory requirements, but in reality `k` is often small enough that it's not a problem.
    // We need both to reach up to `k` as in the worst case all `k` are in exactly one of them.
    // We also need `all_visited` as `l_visited` truncates to `k`, but we also want all visited points in the end.
    // L = l_visited + l_unvisited
    // TODO This is incorrect, as the stopping condition is when the combined L size equals `beam_width`, not just L\V. This means we may be doing way more traversals and stopping way later than necessary.
    let mut l_unvisited = BeamQueue::new(beam_width); // L \ V
    let mut l_visited = BeamQueue::new(beam_width); // V
    let mut all_visited = Bitmap::new();
    l_unvisited.push(PointDist {
      id: start,
      dist: (self.metric)(&self.ds.get_point(start).unwrap().view(), query),
    });
    while let Some(p_star) = l_unvisited.pop() {
      // Move to visited section.
      l_visited.push(p_star);
      all_visited.add(p_star.id);
      for neighbor in self
        .ds
        .get_out_neighbors(p_star.id)
        .unwrap_or_default()
        .into_iter()
      {
        // We separate L out into V and not V, so we must manually ensure the property that l_visited and l_unvisited are disjoint.
        if !all_visited.contains(neighbor) {
          let Some(neighbor_point) = self.ds.get_point(neighbor) else {
            continue;
          };
          l_unvisited.push(PointDist {
            id: neighbor,
            dist: (self.metric)(&neighbor_point.view(), query),
          });
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
    // We may have exceeded k due to pushing both a and b in the last match arm.
    closest.truncate(k);
    (closest, all_visited)
  }

  /// WARNING: `candidates` must not contain the point itself.
  fn compute_robust_pruned(&self, point: &ArrayView1<T>, candidates: Vec<Id>) -> Vec<Id> {
    let dist_thresh = self.params.distance_threshold;
    let degree_bound = self.params.degree_bound;

    let mut candidates = candidates
      .into_par_iter()
      .filter_map(|id| self.ds.get_point(id).map(|p| (id, p)))
      .map(|(id, other_point)| {
        let dist = (self.metric)(&other_point.view(), point);
        (PointDist { id, dist }, other_point)
      })
      .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|s| OrderedFloat(s.0.dist));
    let mut candidates = candidates.into_iter().collect::<VecDeque<_>>();

    let mut new_neighbors = Vec::new();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some((PointDist { id: p_star, .. }, p_star_point)) = candidates.pop_front() {
      new_neighbors.push(p_star);
      if new_neighbors.len() == degree_bound {
        break;
      }
      candidates.retain(|s| {
        let dist_to_p_star = (self.metric)(&s.1.view(), &p_star_point.view());
        dist_thresh * dist_to_p_star > s.0.dist
      });
    }
    new_neighbors
  }

  // The sole purpose of having this separate internal method is the `point_or_is_initialization` parameter, which should only be None from internal use.
  fn _insert(&self, id: Id, point_or_is_initialization: Option<Array1<T>>) {
    // Lock even before RobustPrune, as we don't want a concurrent insert/delete to the same ID while we're processing it (e.g. setting/deleting the point entry in the DB itself).
    let locker = self.update_locker.get(id);
    let _lock = locker.lock();

    let point = match point_or_is_initialization.clone() {
      Some(p) => p,
      None => self.ds.get_point(id).unwrap(),
    };

    // TODO Delete if already exists.

    // Initial GreedySearch.
    let (_closest, mut candidates) =
      self.greedy_search(self.medoid, &point.view(), 1, self.params.beam_width);

    // RobustPrune.
    // RobustPrune requires locking the graph node at this point; we're already holding the lock so we're good to go.
    // Normally, we don't have any existing out-neighbors to add to `candidates`, but on initialization we do (initial R-regular random graph + changes from previous point initializations). Note that it's not safe to fetch the neighbors outside the lock (e.g. pass it into this function) as it may be changed with a crucial edge that we must preserve.
    if point_or_is_initialization.is_none() {
      for n in self.ds.get_out_neighbors(id).unwrap() {
        candidates.add(n);
      }
    }
    // RobustPrune requires that the point itself is never in the candidate set. This should never happen, but let's be safe.
    candidates.remove(id);
    let new_neighbors = self.compute_robust_pruned(&point.view(), candidates.iter().collect());

    // Update neighbors.
    for &j in new_neighbors.iter() {
      // This should never deadlock, as our current point that we're inserting (the outer lock) should not be reachable right now.
      let locker = self.update_locker.get(j);
      let _lock = locker.lock();
      let Some(mut j_neighbors) = self.ds.get_out_neighbors(j) else {
        // Race condition: node has disappeared since.
        continue;
      };
      if j_neighbors.as_slice().contains(&id) {
        // No change necessary.
        continue;
      }
      j_neighbors.push(id);
      if j_neighbors.len() > self.params.degree_bound {
        // It must exist because we acquired the lock and checked it existed.
        let j_point = self.ds.get_point(j).unwrap();
        j_neighbors = self.compute_robust_pruned(&j_point.view(), j_neighbors);
      }
      // Don't batch these outside loop at end, as then we'll hold too many locks for too long and also may cause deadlocks.
      self.ds.set_out_neighbors(j, j_neighbors);
    }

    // If we are initializing, we don't need to set the point.
    if let Some(point) = point_or_is_initialization {
      self.ds.set_point(id, point);
    }
    self.ds.set_out_neighbors(id, new_neighbors);
  }

  pub fn insert(&self, id: Id, point: Array1<T>) {
    self._insert(id, Some(point))
  }

  pub fn query(&self, query: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    self
      .greedy_search(self.medoid, query, k, self.params.beam_width)
      .0
  }
}

#[cfg(test)]
mod tests {
  use super::Vamana;
  use super::VamanaParams;
  use crate::common::metric_euclidean;
  use ahash::AHashMap;
  use croaring::Bitmap;
  use itertools::Itertools;
  use ndarray::array;
  use ndarray::Array;
  use ndarray::Array1;
  use ndarray_rand::RandomExt;
  use ordered_float::OrderedFloat;
  use rand::distributions::Uniform;
  use rand::thread_rng;
  use rand::Rng;
  use serde::Serialize;
  use std::fs::File;
  use std::iter::zip;

  #[test]
  fn test_vamana_2d() {
    let mut rng = thread_rng();
    let metric = metric_euclidean;
    // Let's plot points such that it fits comfortably spread across a widescreen display, useful for when we visualise this.
    let x_range = 0.0f32..1200.0f32;
    let y_range = 0.0f32..700.0f32;
    let n = 100u32;
    let r = 10;
    let ids = (0..n).collect_vec();
    let k = 10;
    let beam_width = k * 2;
    let points = (0..n)
      .map(|_| {
        array![
          rng.gen_range(x_range.clone()),
          rng.gen_range(y_range.clone()),
        ]
      })
      .collect_vec();
    let id_to_point = zip(ids.clone(), points.clone()).collect::<AHashMap<_, _>>();

    let mut vamana = Vamana::new(id_to_point, metric, VamanaParams {
      beam_width,
      degree_bound: r,
      distance_threshold: 1.1,
      medoid_sample_size: 10_000,
    });
    vamana.index();

    // First, test ANN of every point.
    let mut correct = 0;
    for a in ids.iter().cloned() {
      let a_pt = &points[a as usize];
      let truth = ids
        .iter()
        .cloned()
        .filter(|&b| b != a)
        .sorted_unstable_by_key(|&b| OrderedFloat(metric(&a_pt.view(), &points[b as usize].view())))
        .take(k)
        .collect::<Bitmap>();
      let approx = vamana
        .query(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
        .into_iter()
        .map(|pd| pd.id)
        .filter(|&b| b != a)
        .take(k)
        .collect::<Bitmap>();
      correct += approx.and(&truth).cardinality();
    }
    println!(
      "[Pairwise] Correct: {}/{} ({:.2}%)",
      correct,
      k * n as usize,
      correct as f64 / (k * n as usize) as f64 * 100.0
    );

    // Second, test ANN of a query.
    let query = array![rng.gen_range(x_range), rng.gen_range(y_range)];
    let truth = ids
      .iter()
      .cloned()
      .sorted_unstable_by_key(|&id| {
        OrderedFloat(metric(&query.view(), &points[id as usize].view()))
      })
      .take(k)
      .collect::<Bitmap>();
    let approx = vamana
      .query(&query.view(), k)
      .into_iter()
      .map(|pd| pd.id)
      .collect::<Bitmap>();
    let correct = approx.and(&truth).cardinality();
    println!(
      "[Query] Correct: {}/{} ({:.2}%)",
      correct,
      k,
      correct as f64 / k as f64 * 100.0
    );

    #[derive(Serialize)]
    struct DataNode {
      id: u32,
      point: Array1<f32>,
      neighbors: Vec<u32>,
      knn: Vec<u32>,
    }
    let nodes = ids
      .iter()
      .map(|&id| {
        let point = points[id as usize].clone();
        let neighbors = vamana.adj_list[&id].iter().collect();
        let knn = vamana
          .query(&point.view(), k)
          .into_iter()
          .map(|pd| pd.id)
          .collect();
        DataNode {
          id,
          point,
          neighbors,
          knn,
        }
      })
      .collect::<Vec<_>>();

    #[derive(Serialize)]
    struct Data {
      medoid: u32,
      nodes: Vec<DataNode>,
    }

    serde_json::to_writer_pretty(
      File::create("../target/vamana-test-dump.json").unwrap(),
      &Data {
        medoid: vamana.medoid,
        nodes,
      },
    )
    .unwrap();
  }

  #[test]
  fn test_vamana_512d() {
    let metric = metric_euclidean;
    const DIM: usize = 512;
    let n = 1000u32;
    let r = 12;
    let ids = (0..n).collect_vec();
    let k = 15;
    let beam_width = k * 2;

    fn gen_vec() -> Array1<f32> {
      Array::random((DIM,), Uniform::new(-10.0f32, 10.0f32))
    }

    let points = (0..n).map(|_| gen_vec()).collect_vec();
    let id_to_point = zip(ids.clone(), points.clone()).collect::<AHashMap<_, _>>();
    println!("Generated points");

    let mut vamana = Vamana::new(id_to_point, metric, VamanaParams {
      beam_width,
      degree_bound: r,
      distance_threshold: 1.1,
      medoid_sample_size: 1000,
    });
    println!("Initialised");
    vamana.index();
    println!("Indexed");

    // First, test ANN of every point.
    let mut correct = 0;
    for a in ids.iter().cloned() {
      let a_pt = &points[a as usize];
      let truth = ids
        .iter()
        .cloned()
        .filter(|&b| b != a)
        .sorted_unstable_by_key(|&b| OrderedFloat(metric(&a_pt.view(), &points[b as usize].view())))
        .take(k)
        .collect::<Bitmap>();
      let approx = vamana
        .query(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
        .into_iter()
        .map(|pd| pd.id)
        .filter(|&b| b != a)
        .take(k)
        .collect::<Bitmap>();
      correct += approx.and(&truth).cardinality();
    }
    println!(
      "[Pairwise] Correct: {}/{} ({:.2}%)",
      correct,
      k * n as usize,
      correct as f64 / (k * n as usize) as f64 * 100.0
    );

    // Second, test ANN of a query.
    let query = gen_vec();
    let truth = ids
      .iter()
      .cloned()
      .sorted_unstable_by_key(|&id| {
        OrderedFloat(metric(&query.view(), &points[id as usize].view()))
      })
      .take(k)
      .collect::<Bitmap>();
    let approx = vamana
      .query(&query.view(), k)
      .into_iter()
      .map(|pd| pd.id)
      .collect::<Bitmap>();
    let correct = approx.and(&truth).cardinality();
    println!(
      "[Query] Correct: {}/{} ({:.2}%)",
      correct,
      k,
      correct as f64 / k as f64 * 100.0
    );
  }
}
