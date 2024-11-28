use itertools::Itertools;
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::convert::identity;
use std::sync::Arc;
use threadpool::ThreadPool;
use threadpool_scope::scope_with;

// WARNING: Using this like you'd use tokio::spawn_blocking is likely a mistake. We're not async, so running a single task in this thread pool then blocking on it is equivalent to just running the task; we're not yielding so other tasks can work. You almost always want to use IoIteratorExt.
pub(crate) static IO_POOL: Lazy<ThreadPool> = Lazy::new(|| {
  // TODO Tune or allow tuning. Note that NVMe SSDs can do millions of IOPS.
  ThreadPool::new(1024)
});

pub trait IoIteratorExt<T: Send>: Iterator<Item = T> + Sized {
  // Faster if you don't need it ordered (e.g. you'll sort it differently anyway).
  fn io_map_unordered<R, F>(self, f: F) -> impl Iterator<Item = R>
  where
    R: Send,
    F: Fn(T) -> R + Copy + Send,
  {
    let (send, recv) = std::sync::mpsc::channel();
    scope_with(&*IO_POOL, move |scope| {
      for v in self.into_iter() {
        let send = send.clone();
        scope.execute(move || {
          send.send(f(v)).unwrap();
        });
      }
    });
    recv.into_iter()
  }

  fn io_map<R, F>(self, f: F) -> impl Iterator<Item = R>
  where
    R: Send,
    F: Fn(T) -> R + Copy + Send,
  {
    let (send, recv) = std::sync::mpsc::channel();
    scope_with(&*IO_POOL, move |scope| {
      for (i, v) in self.enumerate() {
        let send = send.clone();
        scope.execute(move || {
          send.send((i, f(v))).unwrap();
        });
      }
    });
    recv
      .into_iter()
      .sorted_unstable_by_key(|e| e.0)
      .map(|e| e.1)
  }

  fn io_filter_map<R, F>(self, f: F) -> impl Iterator<Item = R>
  where
    R: Send,
    F: Fn(T) -> Option<R> + Copy + Send,
  {
    let (send, recv) = std::sync::mpsc::channel();
    scope_with(&*IO_POOL, move |scope| {
      for (i, v) in self.enumerate() {
        let send = send.clone();
        scope.execute(move || {
          if let Some(r) = f(v) {
            send.send((i, r)).unwrap();
          };
        });
      }
    });
    recv
      .into_iter()
      .sorted_unstable_by_key(|e| e.0)
      .map(|e| e.1)
  }

  fn io_for_each<F>(self, f: F)
  where
    F: Fn(T) + Copy + Send,
  {
    scope_with(&*IO_POOL, move |scope| {
      // Don't use Barrier, as it will block I/O thread pool threads and can deadlock if there are more tasks than threads.
      for v in self.into_iter() {
        scope.execute(move || {
          f(v);
        });
      }
      scope.join_all();
    });
  }
}

impl<T: Send + 'static, I: Iterator<Item = T>> IoIteratorExt<T> for I {}

#[cfg(test)]
mod tests_ioiteratorext {
  use super::IoIteratorExt;
  use dashmap::DashSet;
  use itertools::Itertools;
  use rand::thread_rng;
  use rand::Rng;
  use std::thread::sleep;
  use std::time::Duration;

  #[test]
  fn test_io_map() {
    let local = DashSet::new();
    let n = 10_000;
    let res = (0..n)
      .io_map(|i| {
        // Simulate out-of-order completion and I/O blocking.
        sleep(Duration::from_micros(thread_rng().gen_range(0..1000)));
        // Do something with the outer scope.
        local.insert(i);
        i * 2
      })
      .collect_vec();
    for i in 0..n {
      assert!(local.contains(&i));
      assert_eq!(res[i], i * 2);
    }
  }

  #[test]
  fn test_io_for_each() {
    let local = DashSet::new();
    let n = 10_000;
    (0..n).io_for_each(|i| {
      // Simulate out-of-order completion and I/O blocking.
      sleep(Duration::from_micros(thread_rng().gen_range(0..1000)));
      // Do something with the outer scope.
      local.insert(i);
    });
    for i in 0..n {
      assert!(local.contains(&i));
    }
  }
}

pub fn insert_into_ordered_vecdeque<T: Clone, K: Ord>(
  dest: &mut VecDeque<T>,
  src: &[T],
  key: impl Fn(&T) -> K,
) {
  for v in src.into_iter() {
    // WARNING: We can't collect all positions and then insert by index descending, as that doesn't account for when multiple source values are inserted at the same position but between themselves are not sorted. This costs the same anyway because we need to do N insertions.
    let pos = dest
      .binary_search_by(|s| key(s).cmp(&key(v)))
      .map_or_else(identity, identity);
    dest.insert(pos, v.clone());
  }
}

#[cfg(test)]
mod tests_insert_into_ordered_vecdeque {
  use crate::util::insert_into_ordered_vecdeque;
  use itertools::Itertools;
  use std::collections::VecDeque;

  #[test]
  fn test_insert_into_ordered_vecdeque() {
    // Empty destination.
    let mut dest = VecDeque::new();
    insert_into_ordered_vecdeque(&mut dest, &[3, 1, 4], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 3, 4]);

    // Empty source.
    let mut dest = VecDeque::from(vec![1, 2, 3]);
    insert_into_ordered_vecdeque(&mut dest, &[], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3]);

    // Both empty.
    let mut dest = VecDeque::<usize>::new();
    insert_into_ordered_vecdeque(&mut dest, &[], |&x| x);
    assert!(dest.is_empty());

    // Interleaved values.
    let mut dest = VecDeque::from(vec![2, 4, 6]);
    insert_into_ordered_vecdeque(&mut dest, &[1, 3, 5], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3, 4, 5, 6]);

    // All duplicates.
    let mut dest = VecDeque::from(vec![1, 1, 1]);
    insert_into_ordered_vecdeque(&mut dest, &[1, 1], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 1, 1, 1, 1]);

    // All source elements larger.
    let mut dest = VecDeque::from(vec![1, 2, 3]);
    insert_into_ordered_vecdeque(&mut dest, &[4, 5, 6], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3, 4, 5, 6]);

    // All source elements smaller.
    let mut dest = VecDeque::from(vec![4, 5, 6]);
    insert_into_ordered_vecdeque(&mut dest, &[1, 2, 3], |&x| x);
    assert_eq!(dest.into_iter().collect_vec(), vec![1, 2, 3, 4, 5, 6]);

    // Custom key function.
    let mut dest = VecDeque::from(vec!["aa", "cccc"]);
    insert_into_ordered_vecdeque(&mut dest, &["bbb", "d"], |s| s.len());
    assert_eq!(dest.into_iter().collect_vec(), vec![
      "d", "aa", "bbb", "cccc"
    ]);

    // Single element source and dest.
    let mut dest = VecDeque::from(vec![2]);
    insert_into_ordered_vecdeque(&mut dest, &[1], |&x| x);
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
