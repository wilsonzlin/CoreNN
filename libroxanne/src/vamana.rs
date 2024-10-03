use crate::beamqueue::BeamQueue;
use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use ahash::AHashMap;
use croaring::Bitmap;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::VecDeque;

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

pub struct Vamana<T: Scalar> {
  adj_list: AHashMap<Id, Bitmap>,
  id_to_point: AHashMap<Id, Array1<T>>,
  metric: Metric<T>,
  medoid: Id,
  params: VamanaParams,
}

impl<T: Scalar> Vamana<T> {
  pub fn new(
    id_to_point: AHashMap<Id, Array1<T>>,
    metric: Metric<T>,
    params: VamanaParams,
  ) -> Self {
    let ids = id_to_point.keys().copied().collect_vec();

    // Initialise to R-regular graph with random edges.
    let adj_list = ids
      .iter()
      .map(|&id| {
        let mut rng = thread_rng();
        let neighbors = ids
          .choose_multiple(&mut rng, params.degree_bound)
          .copied()
          .collect();
        (id, neighbors)
      })
      .collect();

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = {
      let sample_ids = ids
        .choose_multiple(&mut thread_rng(), params.medoid_sample_size)
        .copied()
        .collect_vec();
      let sample_points = sample_ids
        .iter()
        .map(|id| id_to_point[id].view())
        .collect_vec();
      ids
        .iter()
        .copied()
        .min_by_key(|id| {
          let p = id_to_point[id].view();
          OrderedFloat(sample_points.iter().map(|op| metric(&p, op)).sum::<f64>())
        })
        .unwrap()
    };

    Self {
      adj_list,
      id_to_point,
      medoid,
      metric,
      params,
    }
  }

  // DiskANN paper, Algorithm 1: GreedySearch.
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
      dist: (self.metric)(&self.id_to_point[&start].view(), query),
    });
    while let Some(p_star) = l_unvisited.pop() {
      // Move to visited section.
      l_visited.push(p_star);
      all_visited.add(p_star.id);
      for neighbor in self.adj_list[&p_star.id].iter() {
        // We separate L out into V and not V, so we must manually ensure the property that l_visited and l_unvisited are disjoint.
        if !all_visited.contains(neighbor) {
          l_unvisited.push(PointDist {
            id: neighbor,
            dist: (self.metric)(&self.id_to_point[&neighbor].view(), query),
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

  // DiskANN paper, Algorithm 2: RobustPrune.
  fn robust_prune(
    &mut self,
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
      .or(&self.adj_list[&point])
      .iter()
      .filter(|&id| id != point)
      .map(|id| PointDist {
        id,
        dist: (self.metric)(
          &self.id_to_point[&id].view(),
          &self.id_to_point[&point].view(),
        ),
      })
      .sorted_unstable_by_key(|s| OrderedFloat(s.dist))
      .collect::<VecDeque<_>>();

    let point_neighbors = self.adj_list.get_mut(&point).unwrap();
    point_neighbors.clear();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some(PointDist { id: p_star, .. }) = candidates.pop_front() {
      point_neighbors.add(p_star);
      if point_neighbors.cardinality() as usize == degree_bound {
        break;
      }
      candidates.retain(|s| {
        let dist_to_p_star = (self.metric)(
          &self.id_to_point[&s.id].view(),
          &self.id_to_point[&p_star].view(),
        );
        distance_threshold * dist_to_p_star > s.dist
      });
    }
  }

  fn index_pass(&mut self, distance_threshold_override: Option<f64>) {
    let dist_thresh = distance_threshold_override.unwrap_or(self.params.distance_threshold);

    // Iterate points in random order.
    let mut ids_rand = self.id_to_point.keys().copied().collect_vec();
    ids_rand.shuffle(&mut thread_rng());
    for id in ids_rand {
      let (_closest, visited) = self.greedy_search(
        self.medoid,
        &self.id_to_point[&id].view(),
        1,
        self.params.beam_width,
      );
      self.robust_prune(id, visited, dist_thresh, self.params.degree_bound);
      // We will update the graph, so to satisfy Rust, we must clone now. (As a human, I can say that `j` should never equal `id` so this would be safe without cloning, but the compiler doesn't know.)
      for j in self.adj_list[&id].clone().iter() {
        // It's safe to always add `id` and save a clone + add, even though this differs from Algorithm 3 in the paper.
        // - In the first branch, `robust_prune` will immediately clear the neighbors.
        // - In the second branch, we will always add `id`.
        let j_neighbors = self.adj_list.get_mut(&j).unwrap();
        j_neighbors.add(id);
        if j_neighbors.cardinality() as usize > self.params.degree_bound {
          // Clone now instead of as inline arg so that we can drop the mut borrow to adj_list now and therefore mut borrow it in the first arg to robust_prune.
          let candidates = j_neighbors.clone();
          self.robust_prune(j, candidates, dist_thresh, self.params.degree_bound);
        }
      }
    }
  }

  pub fn index(&mut self) {
    // The paper recommends two passes: one with a distance threshold of 1, and one with the user provided distance threshold.
    self.index_pass(Some(1.0));
    self.index_pass(None);
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
