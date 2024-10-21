use crate::common::Id;
use crate::common::Metric;
use crate::common::PointDist;
use ahash::AHashSet;
use dashmap::DashMap;
use dashmap::DashSet;
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
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use std::collections::VecDeque;

// Return owned values:
// - We use DashMap for in-memory, so we can't return a ref while holding a lock in the map entry.
// - From disk, we copy the bytes.
pub trait VamanaDatastore<T: Scalar + Send + Sync>: Sync {
  fn get_point(&self, id: Id) -> Option<Array1<T>>;
  fn set_point(&self, id: Id, point: Array1<T>);
  fn get_out_neighbors(&self, id: Id) -> Option<AHashSet<Id>>;
  fn set_out_neighbors(&self, id: Id, neighbors: AHashSet<Id>);
}

#[derive(Default)]
pub struct InMemoryVamana<T: Scalar + Send + Sync> {
  adj_list: DashMap<Id, AHashSet<Id>>,
  id_to_point: DashMap<Id, Array1<T>>,
}

impl<T: Scalar + Send + Sync> VamanaDatastore<T> for InMemoryVamana<T> {
  fn get_point(&self, id: Id) -> Option<Array1<T>> {
    self.id_to_point.get(&id).map(|e| e.clone())
  }

  fn set_point(&self, id: Id, point: Array1<T>) {
    self.id_to_point.insert(id, point);
  }

  fn get_out_neighbors(&self, id: Id) -> Option<AHashSet<Id>> {
    self.adj_list.get(&id).map(|e| e.clone())
  }

  fn set_out_neighbors(&self, id: Id, neighbors: AHashSet<Id>) {
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
    let adj_list = DashMap::new();
    dataset.par_iter().for_each(|(id, _)| {
      let mut rng = thread_rng();
      let neighbors = dataset
        .choose_multiple(&mut rng, params.degree_bound + 1) // Choose +1 in case we pick self.
        .map(|e| e.0)
        .filter(|oid| id != oid)
        .take(params.degree_bound)
        .collect::<AHashSet<_>>();
      adj_list.insert(*id, neighbors);
    });

    // The medoid will be the starting point `s` as referred in the DiskANN paper (2.3).
    let medoid = {
      let mut rng = thread_rng();
      let sample_nos = (0..params.medoid_sample_size)
        .map(|_| rng.gen_range(0..dataset.len()))
        .collect_vec();
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

    let ids = dataset.iter().map(|e| e.0).collect_vec();
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

    graph.optimize(&ids);

    graph
  }
}

#[derive(Clone)]
pub struct VamanaParams {
  // Calculating the medoid is expensive, so we approximate it via a smaller random sample of points instead.
  pub medoid_sample_size: usize,
  // Must be greater than `k` (according to paper). This corresponds to `L` in the paper.
  pub search_list_cap: usize,
  // Corresponds to `α` in the DiskANN paper. Must be at least 1.
  pub distance_threshold: f64,
  // Corresponds to `R` in the DiskANN paper. The paper recommends at least log(N), where N is the number of points.
  pub degree_bound: usize,
  pub insert_batch_size: usize,
  // Corresponds to W in the DiskANN paper, section 3.3 (DiskANN Beam Search).
  pub beam_width: usize,
}

pub struct Vamana<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> {
  ds: DS,
  medoid: Id,
  metric: Metric<T>,
  params: VamanaParams,
}

impl<T: Scalar + Send + Sync, DS: VamanaDatastore<T>> Vamana<T, DS> {
  pub fn new(ds: DS, medoid: Id, metric: Metric<T>, params: VamanaParams) -> Self {
    Self {
      ds,
      medoid,
      metric,
      params,
    }
  }

  // DiskANN paper, Algorithm 1: GreedySearch.
  // Returns a pair: (closest points, visited node IDs).
  fn greedy_search(&self, query: &ArrayView1<T>, k: usize) -> (Vec<PointDist>, AHashSet<Id>) {
    let start = self.medoid;
    let search_list_cap = self.params.search_list_cap;
    assert!(
      search_list_cap > k,
      "search list capacity must be greater than k"
    );

    // It's too inefficient to calculate L\V repeatedly.
    // Since we need both L (return value) and L\V (each iteration), we split L into V and ¬V.
    // For simplicity, we'll just allow both to reach `k` size, and do a final merge at the end. This doubles the memory requirements, but in reality `k` is often small enough that it's not a problem.
    // We need both to reach up to `k` as in the worst case all `k` are in exactly one of them.
    // We also need `all_visited` as `l_visited` truncates to `k`, but we also want all visited points in the end.
    // L = l_visited + l_unvisited
    let mut l_unvisited = VecDeque::<PointDist>::new(); // L \ V
    let mut l_visited = VecDeque::<PointDist>::new(); // V
    let mut all_visited = AHashSet::new();
    l_unvisited.push_back(PointDist {
      id: start,
      dist: (self.metric)(&self.ds.get_point(start).unwrap().view(), query),
    });
    while !l_unvisited.is_empty() {
      let mut new_visited = (0..self.params.beam_width)
        .filter_map(|_| l_unvisited.pop_front())
        .collect::<VecDeque<_>>();
      let neighbors = DashSet::new();
      new_visited.par_iter().for_each(|p_star| {
        for j in self.ds.get_out_neighbors(p_star.id).unwrap_or_default() {
          neighbors.insert(j);
        }
      });
      // Move to visited section.
      all_visited.extend(new_visited.iter().map(|e| e.id));
      l_visited.append(&mut new_visited);
      l_visited
        .make_contiguous()
        .sort_unstable_by_key(|s| OrderedFloat(s.dist));

      let mut new_unvisited = neighbors
        .into_par_iter()
        .filter_map(|neighbor| {
          // We separate L out into V and not V, so we must manually ensure the property that l_visited and l_unvisited are disjoint.
          if all_visited.contains(&neighbor) {
            return None;
          };
          let Some(neighbor_point) = self.ds.get_point(neighbor) else {
            return None;
          };
          Some(PointDist {
            id: neighbor,
            dist: (self.metric)(&neighbor_point.view(), query),
          })
        })
        .collect::<VecDeque<_>>();
      l_unvisited.append(&mut new_unvisited);
      l_unvisited
        .make_contiguous()
        .sort_unstable_by_key(|s| OrderedFloat(s.dist));

      while l_unvisited.len() + l_visited.len() > search_list_cap {
        let (Some(u), Some(v)) = (l_unvisited.back(), l_visited.back()) else {
          break;
        };
        if u.dist >= v.dist {
          l_unvisited.pop_back();
        } else {
          l_visited.pop_back();
        }
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
    (closest, all_visited)
  }

  /// WARNING: `candidate_ids` must not contain the point itself.
  fn compute_robust_pruned(
    &self,
    point: &ArrayView1<T>,
    candidate_ids: AHashSet<Id>,
  ) -> AHashSet<Id> {
    let dist_thresh = self.params.distance_threshold;
    let degree_bound = self.params.degree_bound;

    // TODO Why does `.into_par_iter()` instead of `.into_iter().par_bridge()` not work?
    let mut candidates = candidate_ids
      .into_iter()
      .par_bridge()
      .filter_map(|id| self.ds.get_point(id).map(|p| (id, p)))
      .map(|(id, other_point)| {
        let dist = (self.metric)(&other_point.view(), point);
        (PointDist { id, dist }, other_point)
      })
      .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|s| OrderedFloat(s.0.dist));
    let mut candidates = VecDeque::from(candidates);

    let mut new_neighbors = AHashSet::new();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some((PointDist { id: p_star, .. }, p_star_point)) = candidates.pop_front() {
      assert!(new_neighbors.insert(p_star));
      if new_neighbors.len() == degree_bound {
        break;
      }
      candidates.retain(|s| {
        let dist_to_p_star = (self.metric)(&p_star_point.view(), &s.1.view());
        dist_thresh * dist_to_p_star > s.0.dist
      });
    }
    new_neighbors
  }

  // The point referenced by each ID should already be inserted into the DB.
  // This is used when inserting, but also during initialization, so this is a separate function from `insert`.
  // WARNING: The graph must be locked (or otherwise not be mutated) while this function is executing, but it is up to the caller's responsibility.
  fn optimize(&self, ids: &[Id]) {
    for batch in ids.chunks(self.params.insert_batch_size) {
      #[derive(Default)]
      struct Update {
        replacement_base: Option<AHashSet<Id>>,
        additional_edges: AHashSet<Id>,
      }
      let updates = DashMap::<Id, Update>::new();

      batch.into_par_iter().for_each(|&id| {
        let point = self.ds.get_point(id).unwrap();

        // TODO Delete if already exists.

        // Initial GreedySearch.
        let mut candidates = self.greedy_search(&point.view(), 1).1;

        // RobustPrune.
        // RobustPrune requires locking the graph node at this point; we're already holding the lock so we're good to go.
        // Normally, we don't have any existing out-neighbors to add to `candidates`, but on initialization we do (initial R-regular random graph + changes from previous point initializations). Note that it's not safe to fetch the neighbors outside the lock (e.g. pass it into this function) as it may be changed with a crucial edge that we must preserve.
        for n in self.ds.get_out_neighbors(id).unwrap_or_default() {
          candidates.insert(n);
        }
        // RobustPrune requires that the point itself is never in the candidate set.
        candidates.remove(&id);
        let new_neighbors = self.compute_robust_pruned(&point.view(), candidates);
        for &j in new_neighbors.iter() {
          updates.entry(j).or_default().additional_edges.insert(id);
        }
        updates.entry(id).or_default().replacement_base = Some(new_neighbors);
      });

      // Update dirty nodes in this batch.
      updates.into_par_iter().for_each(
        |(
          id,
          Update {
            replacement_base,
            additional_edges,
          },
        )| {
          let mut new_neighbors =
            replacement_base.unwrap_or_else(|| self.ds.get_out_neighbors(id).unwrap());
          for j in additional_edges {
            new_neighbors.insert(j);
          }
          if new_neighbors.len() > self.params.degree_bound {
            let point = self.ds.get_point(id).unwrap();
            new_neighbors = self.compute_robust_pruned(&point.view(), new_neighbors);
          };
          self.ds.set_out_neighbors(id, new_neighbors);
        },
      );
    }
  }

  pub fn insert(&self, mut points: Vec<(Id, Array1<T>)>) {
    points.shuffle(&mut thread_rng());
    let ids = points.iter().map(|(id, _)| *id).collect_vec();
    points
      .into_par_iter()
      .for_each(|(id, point)| self.ds.set_point(id, point));
    self.optimize(&ids);
  }

  pub fn query(&self, query: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    self.greedy_search(query, k).0
  }
}

#[cfg(test)]
mod tests {
  use super::VamanaParams;
  use crate::common::metric_euclidean;
  use crate::vamana::InMemoryVamana;
  use ahash::AHashSet;
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
    let search_list_cap = k * 2;
    let points = (0..n)
      .map(|_| {
        array![
          rng.gen_range(x_range.clone()),
          rng.gen_range(y_range.clone()),
        ]
      })
      .collect_vec();
    let dataset = zip(ids.clone(), points.clone()).collect_vec();

    let vamana = InMemoryVamana::init(dataset, metric, VamanaParams {
      beam_width: 1,
      degree_bound: r,
      distance_threshold: 1.1,
      insert_batch_size: 64,
      medoid_sample_size: 10_000,
      search_list_cap,
    });

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
        .collect::<AHashSet<_>>();
      let approx = vamana
        .query(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
        .into_iter()
        .map(|pd| pd.id)
        .filter(|&b| b != a)
        .take(k)
        .collect::<AHashSet<_>>();
      correct += approx.intersection(&truth).count();
    }
    println!(
      "[2D Pairwise] Correct: {}/{} ({:.2}%)",
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
      .collect::<AHashSet<_>>();
    let approx = vamana
      .query(&query.view(), k)
      .into_iter()
      .map(|pd| pd.id)
      .collect::<AHashSet<_>>();
    let correct = approx.intersection(&truth).count();
    println!(
      "[2D Query] Correct: {}/{} ({:.2}%)",
      correct,
      k,
      correct as f64 / k as f64 * 100.0
    );

    #[derive(Serialize)]
    struct DataNode {
      id: u32,
      point: Vec<f32>,
      neighbors: Vec<u32>,
      knn: Vec<u32>,
    }
    let nodes = ids
      .iter()
      .map(|&id| {
        let point = &points[id as usize];
        let neighbors = vamana
          .ds
          .adj_list
          .get(&id)
          .unwrap()
          .iter()
          .copied()
          .collect();
        let knn = vamana
          .query(&point.view(), k)
          .into_iter()
          .map(|pd| pd.id)
          .collect();
        DataNode {
          id,
          point: point.to_vec(),
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
    let search_list_cap = k * 2;

    fn gen_vec() -> Array1<f32> {
      Array::random((DIM,), Uniform::new(-10.0f32, 10.0f32))
    }

    let points = (0..n).map(|_| gen_vec()).collect_vec();
    let dataset = zip(ids.clone(), points.clone()).collect_vec();

    let vamana = InMemoryVamana::init(dataset, metric, VamanaParams {
      beam_width: 1,
      degree_bound: r,
      distance_threshold: 1.1,
      insert_batch_size: 64,
      medoid_sample_size: 1000,
      search_list_cap,
    });

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
        .collect::<AHashSet<_>>();
      let approx = vamana
        .query(&a_pt.view(), k + 1) // +1 because the query point itself should be in the result.
        .into_iter()
        .map(|pd| pd.id)
        .filter(|&b| b != a)
        .take(k)
        .collect::<AHashSet<_>>();
      correct += approx.intersection(&truth).count();
    }
    println!(
      "[512D Pairwise] Correct: {}/{} ({:.2}%)",
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
      .collect::<AHashSet<_>>();
    let approx = vamana
      .query(&query.view(), k)
      .into_iter()
      .map(|pd| pd.id)
      .collect::<AHashSet<_>>();
    let correct = approx.intersection(&truth).count();
    println!(
      "[512D Query] Correct: {}/{} ({:.2}%)",
      correct,
      k,
      correct as f64 / k as f64 * 100.0
    );
  }
}
