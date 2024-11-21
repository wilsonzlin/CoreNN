use ahash::HashMap;
use bitcode::Decode;
use bitcode::Encode;
use bytemuck::Pod;
use dashmap::mapref::one::Ref;
use dashmap::mapref::one::RefMut;
use linfa::Float;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Borrow;
use strum_macros::Display;
use strum_macros::EnumString;

pub type Id = usize;
pub type Metric<T> = fn(&ArrayView1<T>, &ArrayView1<T>) -> f64;

// ndarray_linalg::Scalar required for ndarray.
// linfa::Float required for ProductQuantizer.
pub trait Dtype: Pod + Scalar + Float + Send + Sync + Encode + for<'a> Decode<'a> {}
// TODO Support f16 once ndarray_linalg::Scalar supports it.
impl Dtype for f32 {}
impl Dtype for f64 {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PointDist {
  pub id: Id,
  pub dist: f64,
}

// A metric implementation of the Euclidean distance.
// Casting vectors to f64 will improve precision during intermediate calculations, but might slow down computation (e.g. fp16 can fit more into single AVX512 instruction over fp32). Since it's a tossup, let the caller cast to f64 if they want more precision.
pub fn metric_euclidean<T: Dtype>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let diff = a - b;
  let squared_diff = &diff * &diff;
  let sum_squared_diff = squared_diff.sum();
  sum_squared_diff.to_f64().unwrap().sqrt()
}

// A metric implementation of the cosine distance (NOT similarity).
// Casting vectors to f64 will improve precision during intermediate calculations, but might slow down computation (e.g. fp16 can fit more into single AVX512 instruction over fp32). Since it's a tossup, let the caller cast to f64 if they want more precision.
pub fn metric_cosine<T: Dtype>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let dot_product = a.dot(b).to_f64().unwrap();

  let a_norm = a.dot(a).to_f64().unwrap();
  let b_norm = b.dot(b).to_f64().unwrap();

  let denominator = (a_norm * b_norm).sqrt();

  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product / denominator
  }
}

// Inspired by https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html.
pub fn nan_to_num<T: Dtype>(v: T) -> T {
  if v.is_nan() {
    T::zero()
  } else if v.is_infinite() {
    if v.is_sign_positive() {
      T::max_value()
    } else {
      // WARNING: This is not the same as f32::MIN_VALUE.
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
  pub fn get_fn<T: Dtype>(self) -> fn(&ArrayView1<T>, &ArrayView1<T>) -> f64 {
    match self {
      StdMetric::L2 => metric_euclidean::<T>,
      StdMetric::Cosine => metric_cosine::<T>,
    }
  }
}

pub struct PrecomputedDists {
  id_to_no: HashMap<Id, usize>,
  // Precomputed dists are always f16 on disk, but even when in memory, store as f16 in memory to save memory.
  matrix_flat: Vec<f16>,
}

impl PrecomputedDists {
  pub fn new(id_to_no: HashMap<Id, usize>, matrix_flat: Vec<f16>) -> Self {
    Self {
      id_to_no,
      matrix_flat,
    }
  }

  pub fn get(&self, a: Id, b: Id) -> f64 {
    let n = self.id_to_no.len();
    let ia = self.id_to_no[&a];
    let ib = self.id_to_no[&b];
    self.matrix_flat[ia * n + ib].into()
  }
}

// dashmap::Ref and RefMut do not implement Borrow, only Deref, so we create wrapper newtypes to implement Borrow.

pub struct DashMapValue<'a, V>(pub Ref<'a, Id, V>);

impl<'a, V> Borrow<V> for DashMapValue<'a, V> {
  fn borrow(&self) -> &V {
    self.0.value()
  }
}

pub struct DashMapValueMut<'a, V>(pub RefMut<'a, Id, V>);

impl<'a, V> Borrow<V> for DashMapValueMut<'a, V> {
  fn borrow(&self) -> &V {
    self.0.value()
  }
}
