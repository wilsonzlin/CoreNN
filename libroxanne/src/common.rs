use half::f16;
use ndarray::ArrayView1;
use serde::Deserialize;
use serde::Serialize;
use strum_macros::Display;
use strum_macros::EnumString;

pub type Id = usize;
pub type Metric = fn(&ArrayView1<f16>, &ArrayView1<f16>) -> f32;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PointDist {
  pub id: Id,
  pub dist: f32,
}

pub fn dist_l2(a: &ArrayView1<f16>, b: &ArrayView1<f16>) -> f32 {
  // We must convert to f32 to avoid overflow.
  // It's also what most hardware does optimally.
  let a = a.mapv(f16::to_f32);
  let b = b.mapv(f16::to_f32);
  let diff = a - b;
  let squared_diff = &diff * &diff;
  let sum_squared_diff = squared_diff.sum();
  sum_squared_diff.sqrt()
}

pub fn dist_cos(a: &ArrayView1<f16>, b: &ArrayView1<f16>) -> f32 {
  // We must convert to f32 to avoid overflow.
  // It's also what most hardware does optimally.
  let a = a.mapv(f16::to_f32);
  let b = b.mapv(f16::to_f32);

  let dot_product = a.dot(&b);

  let a_norm = a.dot(&a);
  let b_norm = b.dot(&b);

  let denominator = (a_norm * b_norm).sqrt();

  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product / denominator
  }
}

// Inspired by https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html.
pub fn nan_to_num<T: num::Float>(v: T) -> T {
  if v.is_nan() {
    T::zero()
  } else if v.is_infinite() {
    if v.is_sign_positive() {
      T::max_value()
    } else {
      // WARNING: This is not the same as f32::MIN_VALUE, which is 0.000...
      -T::max_value()
    }
  } else {
    v
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display, EnumString, Serialize, Deserialize)]
pub enum StdMetric {
  L2,
  Cosine,
}

impl StdMetric {
  pub fn get_fn(self) -> fn(&ArrayView1<f16>, &ArrayView1<f16>) -> f32 {
    match self {
      StdMetric::L2 => dist_l2,
      StdMetric::Cosine => dist_cos,
    }
  }
}
