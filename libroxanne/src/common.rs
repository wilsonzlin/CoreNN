use ndarray::ArrayView1;
use ndarray_linalg::Scalar;

// More than 4B is unlikely to scale well on a single shard. Using u32 instead of u64 allows us to use fast RoaringBitmaps everywhere.
pub type Id = u32;
pub type Metric<T> = fn(&ArrayView1<T>, &ArrayView1<T>) -> f64;

#[derive(Clone, Copy, Debug)]
pub struct PointDist {
  pub id: Id,
  pub dist: f64,
}

// A metric implementation of the Euclidean distance.
pub fn metric_euclidean<T: Scalar>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let diff = a - b;
  let squared_diff = &diff * &diff;
  let sum_squared_diff = squared_diff.sum();
  sum_squared_diff.to_f64().unwrap().sqrt()
}

// A metric implementation of the cosine distance (NOT similarity).
pub fn metric_cosine<T: Scalar>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
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
