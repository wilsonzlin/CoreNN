use futures::stream::FuturesOrdered;
use futures::stream::FuturesUnordered;
use futures::Stream;
use futures::StreamExt;
use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub fn unarc<T>(arc: Arc<T>) -> T {
  let Some(v) = Arc::into_inner(arc) else {
    panic!("Arc is not owned");
  };
  v
}

pub trait AsyncConcurrentIteratorExt<T>: Iterator<Item = T> + Sized {
  /// Faster than map_concurrent if strict FIFO ordering is unnecessary.
  /// Concurrency != parallelism; use this if primarily I/O bound and not CPU.
  fn map_concurrent_unordered<R, Fut, F>(self, f: F) -> impl Stream<Item = R>
  where
    Fut: Future<Output = R>,
    F: Fn(T) -> Fut,
  {
    self.map(|t| f(t)).collect::<FuturesUnordered<_>>()
  }

  /// Concurrency != parallelism; use this if primarily I/O bound and not CPU.
  fn map_concurrent<R, Fut, F>(self, f: F) -> impl Stream<Item = R>
  where
    Fut: Future<Output = R>,
    F: Fn(T) -> Fut,
  {
    // No need for for_each_concurrent, FuturesOrdered "will race [futures] to completion in parallel": https://docs.rs/futures/latest/futures/stream/struct.FuturesOrdered.html.
    self.map(|t| f(t)).collect::<FuturesOrdered<_>>()
  }

  /// Concurrency != parallelism; use this if primarily I/O bound and not CPU.
  fn filter_map_concurrent<R, Fut, F>(self, f: F) -> impl Stream<Item = R>
  where
    Fut: Future<Output = Option<R>>,
    F: Fn(T) -> Fut,
  {
    self
      .map(|t| f(t))
      .collect::<FuturesOrdered<_>>()
      .filter_map(async |x| x)
  }

  /// Convenience method for spawning async tasks to run in the background (for parallelism) and waiting for them to complete. Better than for_each_concurrent, since this is actually parallel.
  /// Since this uses tokio::spawn, the async task must be 'static. Given this is an iterator method, a convenient pattern is to chain a .map *before* this method to collect all args into 'static (e.g. cloning Arcs, cloning into owned values).
  async fn spawn_for_each<Fut, F>(self, f: F)
  where
    Fut: Future<Output = ()> + Send + 'static,
    F: Fn(T) -> Fut,
  {
    self
      .map(|t| tokio::spawn(f(t)))
      .collect::<FuturesUnordered<_>>()
      .for_each(async |e| e.unwrap())
      .await;
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

// AtomicUsize counter with more ergonomic methods.
#[derive(Debug, Default)]
pub struct AtomUsz(AtomicUsize);

impl AtomUsz {
  pub fn new(v: usize) -> Self {
    Self(AtomicUsize::new(v))
  }

  pub fn set(&self, v: usize) {
    self.0.store(v, Ordering::Relaxed);
  }

  // Can't overload += as that's &mut.
  pub fn inc(&self, v: usize) {
    self.0.fetch_add(v, Ordering::Relaxed);
  }

  // Can't overload -= as that's &mut.
  pub fn dec(&self, v: usize) {
    self.0.fetch_sub(v, Ordering::Relaxed);
  }

  // Can't overload Deref as that requires returning a reference.
  pub fn get(&self) -> usize {
    self.0.load(Ordering::Relaxed)
  }
}

impl From<usize> for AtomUsz {
  fn from(v: usize) -> Self {
    Self::new(v)
  }
}
