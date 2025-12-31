use crate::LabelType;

pub trait Space: Clone + Send + Sync + 'static {
  fn dim(&self) -> usize;
  fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

type DistanceFn = unsafe fn(*const f32, *const f32, usize) -> f32;

unsafe fn l2_scalar(a: *const f32, b: *const f32, dim: usize) -> f32 {
  let mut res = 0.0_f32;
  for i in 0..dim {
    let t = *a.add(i) - *b.add(i);
    res += t * t;
  }
  res
}

unsafe fn ip_distance_scalar(a: *const f32, b: *const f32, dim: usize) -> f32 {
  let mut dot = 0.0_f32;
  for i in 0..dim {
    dot += *a.add(i) * *b.add(i);
  }
  1.0_f32 - dot
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_simd {
  use super::DistanceFn;
  #[cfg(target_arch = "x86")]
  use std::arch::x86::*;
  #[cfg(target_arch = "x86_64")]
  use std::arch::x86_64::*;

  #[target_feature(enable = "sse")]
  pub unsafe fn l2_sse(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= dim {
      let va = _mm_loadu_ps(a.add(i));
      let vb = _mm_loadu_ps(b.add(i));
      let diff = _mm_sub_ps(va, vb);
      sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
      i += 4;
    }

    let mut tmp = [0.0_f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut res = tmp.iter().sum::<f32>();

    while i < dim {
      let t = *a.add(i) - *b.add(i);
      res += t * t;
      i += 1;
    }
    res
  }

  #[target_feature(enable = "avx")]
  pub unsafe fn l2_avx(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= dim {
      let va = _mm256_loadu_ps(a.add(i));
      let vb = _mm256_loadu_ps(b.add(i));
      let diff = _mm256_sub_ps(va, vb);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
      i += 8;
    }

    let mut tmp = [0.0_f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut res = tmp.iter().sum::<f32>();

    while i < dim {
      let t = *a.add(i) - *b.add(i);
      res += t * t;
      i += 1;
    }
    res
  }

  #[target_feature(enable = "avx512f")]
  pub unsafe fn l2_avx512(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= dim {
      let va = _mm512_loadu_ps(a.add(i));
      let vb = _mm512_loadu_ps(b.add(i));
      let diff = _mm512_sub_ps(va, vb);
      sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
      i += 16;
    }

    let mut res = _mm512_reduce_add_ps(sum);
    while i < dim {
      let t = *a.add(i) - *b.add(i);
      res += t * t;
      i += 1;
    }
    res
  }

  #[target_feature(enable = "sse")]
  pub unsafe fn ip_distance_sse(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= dim {
      let va = _mm_loadu_ps(a.add(i));
      let vb = _mm_loadu_ps(b.add(i));
      sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
      i += 4;
    }

    let mut tmp = [0.0_f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut dot = tmp.iter().sum::<f32>();

    while i < dim {
      dot += *a.add(i) * *b.add(i);
      i += 1;
    }
    1.0_f32 - dot
  }

  #[target_feature(enable = "avx")]
  pub unsafe fn ip_distance_avx(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= dim {
      let va = _mm256_loadu_ps(a.add(i));
      let vb = _mm256_loadu_ps(b.add(i));
      sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
      i += 8;
    }

    let mut tmp = [0.0_f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut dot = tmp.iter().sum::<f32>();

    while i < dim {
      dot += *a.add(i) * *b.add(i);
      i += 1;
    }
    1.0_f32 - dot
  }

  #[target_feature(enable = "avx512f")]
  pub unsafe fn ip_distance_avx512(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= dim {
      let va = _mm512_loadu_ps(a.add(i));
      let vb = _mm512_loadu_ps(b.add(i));
      sum = _mm512_add_ps(sum, _mm512_mul_ps(va, vb));
      i += 16;
    }

    let mut dot = _mm512_reduce_add_ps(sum);
    while i < dim {
      dot += *a.add(i) * *b.add(i);
      i += 1;
    }
    1.0_f32 - dot
  }

  pub fn pick_l2() -> Option<DistanceFn> {
    if std::is_x86_feature_detected!("avx512f") {
      return Some(l2_avx512);
    }
    if std::is_x86_feature_detected!("avx") {
      return Some(l2_avx);
    }
    if std::is_x86_feature_detected!("sse") {
      return Some(l2_sse);
    }
    None
  }

  pub fn pick_ip_distance() -> Option<DistanceFn> {
    if std::is_x86_feature_detected!("avx512f") {
      return Some(ip_distance_avx512);
    }
    if std::is_x86_feature_detected!("avx") {
      return Some(ip_distance_avx);
    }
    if std::is_x86_feature_detected!("sse") {
      return Some(ip_distance_sse);
    }
    None
  }
}

#[derive(Clone, Debug)]
pub struct L2Space {
  dim: usize,
  dist_fn: DistanceFn,
}

impl L2Space {
  pub fn new(dim: usize) -> Self {
    let mut dist_fn: DistanceFn = l2_scalar;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(f) = x86_simd::pick_l2() {
      dist_fn = f;
    }
    Self { dim, dist_fn }
  }
}

impl Space for L2Space {
  fn dim(&self) -> usize {
    self.dim
  }

  fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), self.dim);
    debug_assert_eq!(b.len(), self.dim);
    unsafe { (self.dist_fn)(a.as_ptr(), b.as_ptr(), self.dim) }
  }
}

#[derive(Clone, Debug)]
pub struct InnerProductSpace {
  dim: usize,
  dist_fn: DistanceFn,
}

impl InnerProductSpace {
  pub fn new(dim: usize) -> Self {
    let mut dist_fn: DistanceFn = ip_distance_scalar;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(f) = x86_simd::pick_ip_distance() {
      dist_fn = f;
    }
    Self { dim, dist_fn }
  }
}

impl Space for InnerProductSpace {
  fn dim(&self) -> usize {
    self.dim
  }

  fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), self.dim);
    debug_assert_eq!(b.len(), self.dim);
    unsafe { (self.dist_fn)(a.as_ptr(), b.as_ptr(), self.dim) }
  }
}

pub fn normalize_cosine_in_place(vector: &mut [f32]) {
  let mut norm_sq = 0.0_f32;
  for &v in vector.iter() {
    norm_sq += v * v;
  }
  if norm_sq == 0.0 {
    return;
  }
  let inv_norm = norm_sq.sqrt().recip();
  for v in vector.iter_mut() {
    *v *= inv_norm;
  }
}

pub fn label_allowed(filter: Option<&dyn Fn(LabelType) -> bool>, label: LabelType) -> bool {
  filter.map(|f| f(label)).unwrap_or(true)
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;
  use rand::rngs::StdRng;
  use rand::Rng;
  use rand::SeedableRng;

  fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
      .zip(b.iter())
      .map(|(a, b)| {
        let d = a - b;
        d * d
      })
      .sum()
  }

  fn ip_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>()
  }

  #[test]
  fn l2_distance_matches_scalar_with_simd_dispatch() {
    let mut rng = StdRng::seed_from_u64(123);
    let dims = [
      1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    ];
    for &dim in &dims {
      let space = L2Space::new(dim);
      for _ in 0..100 {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let expected = l2_scalar(&a, &b);
        let got = space.distance(&a, &b);
        assert_relative_eq!(got, expected, epsilon = 1e-3, max_relative = 1e-3);
      }
    }
  }

  #[test]
  fn inner_product_distance_matches_scalar_with_simd_dispatch() {
    let mut rng = StdRng::seed_from_u64(456);
    let dims = [
      1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    ];
    for &dim in &dims {
      let space = InnerProductSpace::new(dim);
      for _ in 0..100 {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let expected = ip_distance_scalar(&a, &b);
        let got = space.distance(&a, &b);
        assert_relative_eq!(got, expected, epsilon = 1e-3, max_relative = 1e-3);
      }
    }
  }
}
