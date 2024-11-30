use ahash::HashMap;
use bytemuck::Pod;
use linfa::Float;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray_linalg::Scalar;
use num::NumCast;
use serde::Deserialize;
use serde::Serialize;
use strum_macros::Display;
use strum_macros::EnumString;

pub type Id = usize;
pub type Metric<T> = fn(&ArrayView1<T>, &ArrayView1<T>) -> f64;

// bytemuck::Pod required for bytemuck.
// num::NumCast required for casting to DtypeCalc.
// num::Float required for nan_to_num.
pub trait Dtype: Pod + Send + Sync + NumCast + num::Float {}
impl Dtype for half::f16 {}
impl Dtype for f32 {}
impl Dtype for f64 {}

// We use a separate trait for calculations as f16 is not supported by crucial libraries: ndarray, num_traits, linfa. Therefore, Dtype represents data stored on disk and memory, and vectors in Dtype will be converted to DtypeCalc before any calculations, allowing us to retain the benefit of the space efficiency f16 in memory and on disk.
// ndarray_linalg::Scalar required for ndarray.
// linfa::Float required for ProductQuantizer.
pub trait DtypeCalc: Scalar + Float {}
impl DtypeCalc for f32 {}
impl DtypeCalc for f64 {}

pub fn to_calc<T: Dtype, C: DtypeCalc>(v: &ArrayView1<T>) -> Array1<C> {
  v.mapv(|t| num::cast::<T, C>(t).unwrap())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PointDist {
  pub id: Id,
  pub dist: f64,
}

// A metric implementation of the Euclidean distance.
// Casting vectors to f64 will improve precision during intermediate calculations, but might slow down computation (e.g. fp16 can fit more into single AVX512 instruction over fp32). Since it's a tossup, let the caller cast to f64 if they want more precision.
pub fn metric_euclidean<C: DtypeCalc>(a: &ArrayView1<C>, b: &ArrayView1<C>) -> f64 {
  let diff = a - b;
  let squared_diff = &diff * &diff;
  let sum_squared_diff = squared_diff.sum();
  sum_squared_diff.to_f64().unwrap().sqrt()
}

// A metric implementation of the cosine distance (NOT similarity).
// Casting vectors to f64 will improve precision during intermediate calculations, but might slow down computation (e.g. fp16 can fit more into single AVX512 instruction over fp32). Since it's a tossup, let the caller cast to f64 if they want more precision.
pub fn metric_cosine<C: DtypeCalc>(a: &ArrayView1<C>, b: &ArrayView1<C>) -> f64 {
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
pub fn nan_to_num<T: num::Float>(v: T) -> T {
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
  pub fn get_fn<T: DtypeCalc>(self) -> fn(&ArrayView1<T>, &ArrayView1<T>) -> f64 {
    match self {
      StdMetric::L2 => metric_euclidean::<T>,
      StdMetric::Cosine => metric_cosine::<T>,
    }
  }
}

pub struct PrecomputedDists {
  id_to_no: HashMap<Id, usize>,
  // Precomputed dists are always f16 on disk, but even when in memory, store as f16 in memory to save memory.
  matrix_flat: Vec<half::f16>,
}

impl PrecomputedDists {
  pub fn new(id_to_no: HashMap<Id, usize>, matrix_flat: Vec<half::f16>) -> Self {
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
