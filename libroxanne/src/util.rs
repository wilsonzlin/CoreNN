use futures::stream::FuturesOrdered;
use futures::stream::FuturesUnordered;
use futures::Stream;
use futures::StreamExt;
use std::collections::VecDeque;
use std::convert::identity;
use std::future::Future;
use std::sync::Arc;

pub trait CollectionExt<T> {
  fn insert_into_ordered<K: Ord>(&mut self, v: T, key: impl Fn(&T) -> K);

  // WARNING: We can't collect all positions and then insert by index descending, as that doesn't account for when multiple source values are inserted at the same position but between themselves are not sorted. This costs the same anyway because we need to do N insertions.
  fn extend_into_ordered<K: Ord>(
    &mut self,
    src: impl IntoIterator<Item = T>,
    key: impl Fn(&T) -> K + Copy,
  ) {
    for v in src {
      self.insert_into_ordered(v, key);
    }
  }
}

impl<T> CollectionExt<T> for VecDeque<T> {
  fn insert_into_ordered<K: Ord>(&mut self, v: T, key: impl Fn(&T) -> K) {
    let pos = self
      .binary_search_by(|s| key(s).cmp(&key(&v)))
      .map_or_else(identity, identity);
    self.insert(pos, v);
  }
}

#[cfg(test)]
mod tests_insert_into_ordered_vecdeque {
  use crate::util::CollectionExt;
  use itertools::Itertools;
  use std::collections::VecDeque;

  #[test]
  fn test_insert_into_ordered_vecdeque() {
    // Empty destination.
    let mut dest = VecDeque::new();
    dest.extend_into_ordered([3, 1, 4], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 3, 4]);

    // Empty source.
    let mut dest = VecDeque::from(vec![1, 2, 3]);
    dest.extend_into_ordered([], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3]);

    // Both empty.
    let mut dest = VecDeque::<usize>::new();
    dest.extend_into_ordered([], |&x| x);
    assert!(dest.is_empty());

    // Interleaved values.
    let mut dest = VecDeque::from(vec![2, 4, 6]);
    dest.extend_into_ordered([1, 3, 5], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3, 4, 5, 6]);

    // All duplicates.
    let mut dest = VecDeque::from(vec![1, 1, 1]);
    dest.extend_into_ordered([1, 1], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 1, 1, 1, 1]);

    // All source elements larger.
    let mut dest = VecDeque::from(vec![1, 2, 3]);
    dest.extend_into_ordered([4, 5, 6], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3, 4, 5, 6]);

    // All source elements smaller.
    let mut dest = VecDeque::from(vec![4, 5, 6]);
    dest.extend_into_ordered([1, 2, 3], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3, 4, 5, 6]);

    // Custom key function.
    let mut dest = VecDeque::from(vec!["aa", "cccc"]);
    dest.extend_into_ordered(["bbb", "d"], |s| s.len());
    assert_eq!(dest.into_iter().collect_vec(), vec![
      "d", "aa", "bbb", "cccc"
    ]);

    // Single element source and dest.
    let mut dest = VecDeque::from(vec![2]);
    dest.extend_into_ordered([1], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2]);
  }
}

pub struct ArcMap<T, R> {
  _owner: Arc<T>,
  v: R,
}

impl<T, R> ArcMap<T, R> {
  pub fn map<'l, F>(arc: Arc<T>, mapper: F) -> Self
  where
    F: FnOnce(&'l T) -> R,
    T: 'l,
  {
    // SAFETY: Arc allocates on heap and data will always live at that stable address.
    let t_ptr = Arc::as_ptr(&arc);
    let r = mapper(unsafe { &*t_ptr });
    Self { _owner: arc, v: r }
  }
}

impl<T, R> std::ops::Deref for ArcMap<T, R> {
  type Target = R;

  fn deref(&self) -> &Self::Target {
    &self.v
  }
}

/// Concurrency != parallelism; use AsyncParIteratorExt for parallelism. Use this if primarily I/O bound and not CPU; note that AsyncPar* is unsafe right now.
pub trait AsyncConcurrentIteratorExt<T>: Iterator<Item = T> + Sized {
  /// Faster than map_concurrent if strict FIFO ordering is unnecessary.
  fn map_concurrent_unordered<R, Fut, F>(self, f: F) -> impl Stream<Item = R>
  where
    Fut: Future<Output = R>,
    F: Fn(T) -> Fut,
  {
    self.map(|t| f(t)).collect::<FuturesUnordered<_>>()
  }

  fn map_concurrent<R, Fut, F>(self, f: F) -> impl Stream<Item = R>
  where
    Fut: Future<Output = R>,
    F: Fn(T) -> Fut,
  {
    // No need for for_each_concurrent, FuturesOrdered "will race [futures] to completion in parallel": https://docs.rs/futures/latest/futures/stream/struct.FuturesOrdered.html.
    self.map(|t| f(t)).collect::<FuturesOrdered<_>>()
  }
}

impl<T, I: Iterator<Item = T>> AsyncConcurrentIteratorExt<T> for I {}

pub trait AsyncConcurrentStreamExt<T>: Stream<Item = T> + Sized + Unpin {
  /// Useful for providing a closure that mutates local scope variables, which for_each doesn't support.
  async fn for_each_sync(mut self, mut f: impl FnMut(T)) {
    while let Some(v) = self.next().await {
      f(v);
    }
  }

  async fn collect_vec(self) -> Vec<T> {
    self.collect::<Vec<T>>().await
  }
}

impl<T, S: Stream<Item = T> + Unpin> AsyncConcurrentStreamExt<T> for S {}
