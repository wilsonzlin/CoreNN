use ahash::HashSet;
use ahash::HashSetExt;
use libroxanne_search::Id;
use libroxanne_search::PointDist;
use std::collections::VecDeque;

// It's too inefficient to calculate argmin(dist(P, query)) repeatedly.
// Instead, we'll use a sorted array for O(1) argmin.
// We don't use a BinaryHeap as a priority queue does not allow for efficient truncation, which we do at each iteration. A Vec may be slower in theory to insert, but `k` is often small enough that it's actually faster in reality due to cache locality.
// This will also ensure there are no duplicates. (Important, as the algorithm in the DiskANN paper specifies that this is a set.)
pub struct BoundedDistinctQueue {
  queue: VecDeque<PointDist>,
  set: HashSet<Id>,
  k: usize,
}

impl BoundedDistinctQueue {
  pub fn new(k: usize) -> Self {
    Self {
      queue: VecDeque::new(),
      set: HashSet::new(),
      k,
    }
  }

  pub fn len(&self) -> usize {
    self.queue.len()
  }

  pub fn push(&mut self, state: PointDist) {
    debug_assert!(self.queue.len() <= self.k);
    debug_assert!(self.queue.len() == self.set.len());
    // Do not insert if already exists.
    if self.set.contains(&state.id) {
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
    self.set.insert(state.id);
    self.queue.insert(pos, state);
    if self.queue.len() > self.k {
      let PointDist { id, .. } = self.queue.pop_back().unwrap();
      self.set.remove(&id);
    }
  }

  pub fn pop(&mut self) -> Option<PointDist> {
    let s = self.queue.pop_front();
    if let Some(s) = s.as_ref() {
      self.set.remove(&s.id);
    }
    s
  }
}
