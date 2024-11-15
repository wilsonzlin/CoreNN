use crate::common::Dtype;
use crate::common::Id;
use crate::common::PointDist;
use crate::search::GreedySearchable;
use crate::search::Query;
use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashMap;
use itertools::Itertools;
use ndarray::Array1;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Borrow;
use std::collections::VecDeque;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VamanaParams {
  pub update_search_list_cap: usize,
  pub update_batch_size: usize,
  // Corresponds to `α` in the DiskANN paper. Must be at least 1.
  pub distance_threshold: f64,
  // Corresponds to `R` in the DiskANN paper. The paper recommends at least log(N), where N is the number of points.
  pub degree_bound: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizeMetrics {
  pub updated_nodes: HashSet<Id>,
}

pub trait Vamana<'a, T: Dtype>: GreedySearchable<'a, T> + Send + Sync {
  fn params(&self) -> &VamanaParams;
  fn set_point(&self, id: Id, point: Array1<T>);
  fn set_out_neighbors(&self, id: Id, neighbors: Vec<Id>);

  /// WARNING: `candidate_ids` must not contain the point itself.
  fn compute_robust_pruned(
    &'a self,
    node_id: Id,
    candidate_ids: impl IntoIterator<Item = Id>,
  ) -> Vec<Id> {
    let dist_thresh = self.params().distance_threshold;
    let degree_bound = self.params().degree_bound;

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
  fn optimize(
    &'a self,
    mut ids: Vec<Id>,
    mut metrics: Option<&mut OptimizeMetrics>,
    on_progress: impl Fn(usize, Option<&OptimizeMetrics>),
  ) {
    // Shuffle to reduce chance of inserting around the same area in latent space.
    ids.shuffle(&mut thread_rng());
    let mut completed = 0;
    for batch in ids.chunks(self.params().update_batch_size) {
      #[derive(Default)]
      struct Update {
        // These two aren't the same and can't be merged, as otherwise we can't tell whether we are supposed to replace or merge with the existing out-neighbors.
        replacement_base: Option<Vec<Id>>,
        additional_edges: HashSet<Id>,
      }
      let updates = DashMap::<Id, Update>::new();

      batch.into_par_iter().for_each(|&id| {
        // Initial GreedySearch.
        let mut candidates = HashSet::new();
        self.greedy_search(
          Query::Id(id),
          1,
          self.params().update_search_list_cap,
          1,
          self.medoid(),
          |_| true,
          Some(&mut candidates),
          None,
          None,
        );

        // RobustPrune.
        // RobustPrune requires locking the graph node at this point; we're already holding the lock so we're good to go.
        for &n in self.get_out_neighbors(id).0.borrow() {
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
          .unwrap_or_else(|| self.get_out_neighbors(id).0.borrow().clone());
        for j in u.additional_edges {
          if !new_neighbors.contains(&j) {
            new_neighbors.push(j);
          };
        }
        if new_neighbors.len() > self.params().degree_bound {
          new_neighbors = self.compute_robust_pruned(id, new_neighbors);
        };
        self.set_out_neighbors(id, new_neighbors);
      });
      completed += batch.len();
      on_progress(completed, metrics.as_ref().map(|m| &**m));
    }
  }
}
