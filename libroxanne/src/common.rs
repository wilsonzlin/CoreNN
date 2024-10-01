use std::iter::zip;

// More than 4B is unlikely to scale well on a single shard. Using u32 instead of u64 allows us to use fast RoaringBitmaps everywhere.
pub type Id = u32;
pub type Metric<P> = fn(&P, &P) -> f64;

#[derive(Clone, Copy, Debug)]
pub struct PointDist {
  pub id: Id,
  pub dist: f64,
}

// A metric implementation of the Euclidean distance.
pub fn metric_euclidean<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f64 {
  zip(a, b)
    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
    .sum::<f64>()
    .sqrt()
}

// A metric implementation of the cosine similarity.
pub fn metric_cosine<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f64 {
  let dot = zip(a, b).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>();
  let norm_a = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
  let norm_b = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
  1.0 - dot / (norm_a * norm_b)
}
