use crate::scalar::Scalar;
use corenn_kernels::cosine_distance;
use corenn_kernels::inner_product_distance;
use corenn_kernels::Kernel;
use std::marker::PhantomData;

pub trait Metric: Clone + Send + Sync + 'static {
  type Scalar: Scalar;
  fn distance(&self, a: &[Self::Scalar], b: &[Self::Scalar]) -> f32;
}

#[derive(Clone, Debug, Default)]
pub struct L2<S: Scalar = f32>(PhantomData<S>);

impl<S: Scalar> L2<S> {
  pub fn new() -> Self {
    Self(PhantomData)
  }
}

impl<S: Scalar> Metric for L2<S> {
  type Scalar = S;

  fn distance(&self, a: &[S], b: &[S]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    <S as Kernel>::l2_sq(a, b)
  }
}

#[derive(Clone, Debug, Default)]
pub struct InnerProduct<S: Scalar = f32>(PhantomData<S>);

impl<S: Scalar> InnerProduct<S> {
  pub fn new() -> Self {
    Self(PhantomData)
  }
}

impl<S: Scalar> Metric for InnerProduct<S> {
  type Scalar = S;

  fn distance(&self, a: &[S], b: &[S]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    inner_product_distance::<S>(a, b)
  }
}

#[derive(Clone, Debug, Default)]
pub struct Cosine<S: Scalar = f32>(PhantomData<S>);

impl<S: Scalar> Cosine<S> {
  pub fn new() -> Self {
    Self(PhantomData)
  }
}

impl<S: Scalar> Metric for Cosine<S> {
  type Scalar = S;

  fn distance(&self, a: &[S], b: &[S]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    cosine_distance::<S>(a, b)
  }
}

pub fn normalize_cosine_in_place<S: Scalar>(vector: &mut [S]) {
  let mut norm_sq = 0.0_f32;
  for &v in vector.iter() {
    let v = v.to_f32();
    norm_sq += v * v;
  }
  if norm_sq == 0.0 {
    return;
  }
  let inv_norm = norm_sq.sqrt().recip();
  for v in vector.iter_mut() {
    *v = S::from_f32(v.to_f32() * inv_norm);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;
  use rand::rngs::StdRng;
  use rand::Rng;
  use rand::SeedableRng;

  fn l2_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
      .zip(b.iter())
      .map(|(a, b)| {
        let d = a - b;
        d * d
      })
      .sum()
  }

  fn ip_ref(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>()
  }

  #[test]
  fn l2_matches_scalar_across_dims() {
    let mut rng = StdRng::seed_from_u64(123);
    let dims = [
      1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    ];
    let metric = L2::<f32>::new();
    for &dim in &dims {
      for _ in 0..100 {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        assert_relative_eq!(
          metric.distance(&a, &b),
          l2_ref(&a, &b),
          epsilon = 1e-3,
          max_relative = 1e-3
        );
      }
    }
  }

  #[test]
  fn inner_product_matches_scalar_across_dims() {
    let mut rng = StdRng::seed_from_u64(456);
    let dims = [
      1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    ];
    let metric = InnerProduct::<f32>::new();
    for &dim in &dims {
      for _ in 0..100 {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        assert_relative_eq!(
          metric.distance(&a, &b),
          ip_ref(&a, &b),
          epsilon = 1e-3,
          max_relative = 1e-3
        );
      }
    }
  }
}
