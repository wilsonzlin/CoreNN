use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

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
  pub fn add(&self, v: usize) -> usize {
    let old = self.0.fetch_add(v, Ordering::Relaxed);
    old
  }

  pub fn inc(&self) -> usize {
    self.add(1)
  }

  // Can't overload -= as that's &mut.
  pub fn sub(&self, v: usize) -> usize {
    let old = self.0.fetch_sub(v, Ordering::Relaxed);
    old
  }

  pub fn dec(&self) -> usize {
    self.sub(1)
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
