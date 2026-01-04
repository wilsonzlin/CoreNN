#![feature(f16)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_f16))]
#![cfg_attr(
  any(target_arch = "x86", target_arch = "x86_64"),
  feature(stdarch_x86_avx512_f16)
)]

use half::bf16;
use half::f16;

pub trait Kernel: Copy + Send + Sync + 'static {
  fn dot(a: &[Self], b: &[Self]) -> f32;
  fn l2_sq(a: &[Self], b: &[Self]) -> f32;
  fn dot_and_norms(a: &[Self], b: &[Self]) -> (f32, f32, f32);
}

#[derive(Clone, Copy, Debug)]
struct I8Sums {
  dot: i32,
  sum_a: i32,
  sum_b: i32,
  sum_sq_a: i32,
  sum_sq_b: i32,
}

pub fn cosine_distance<T: Kernel>(a: &[T], b: &[T]) -> f32 {
  let (dot, norm_sq_a, norm_sq_b) = T::dot_and_norms(a, b);
  if norm_sq_a == 0.0 || norm_sq_b == 0.0 {
    return 1.0;
  }
  let denom = (norm_sq_a * norm_sq_b).sqrt();
  if denom == 0.0 {
    1.0
  } else {
    1.0 - dot / denom
  }
}

pub fn inner_product_distance<T: Kernel>(a: &[T], b: &[T]) -> f32 {
  1.0 - T::dot(a, b)
}

fn i8_sums_scalar(a: &[i8], b: &[i8]) -> I8Sums {
  debug_assert_eq!(a.len(), b.len());

  let mut dot = 0i32;
  let mut sum_a = 0i32;
  let mut sum_b = 0i32;
  let mut sum_sq_a = 0i32;
  let mut sum_sq_b = 0i32;

  for (&a, &b) in a.iter().zip(b) {
    let a = a as i32;
    let b = b as i32;
    dot += a * b;
    sum_a += a;
    sum_b += b;
    sum_sq_a += a * a;
    sum_sq_b += b * b;
  }

  I8Sums {
    dot,
    sum_a,
    sum_b,
    sum_sq_a,
    sum_sq_b,
  }
}

fn i8_sums(a: &[i8], b: &[i8]) -> I8Sums {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx2") {
      // SAFETY: guarded by runtime feature check.
      unsafe { return x86::i8_sums_avx2(a, b) };
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    // AArch64 always has Neon.
    // SAFETY: intrinsic usage.
    unsafe { return aarch64::i8_sums_neon(a, b) };
  }
  i8_sums_scalar(a, b)
}

pub fn dot_and_norms_qi8(
  a: &[i8],
  a_scale: f32,
  a_zero_point: i8,
  b: &[i8],
  b_scale: f32,
  b_zero_point: i8,
) -> (f32, f32, f32) {
  assert_eq!(a.len(), b.len());

  let sums = i8_sums(a, b);
  let d = a.len() as i32;
  let za = a_zero_point as i32;
  let zb = b_zero_point as i32;

  let ab = sums.dot - zb * sums.sum_a - za * sums.sum_b + d * za * zb;
  let a2 = sums.sum_sq_a - 2 * za * sums.sum_a + d * za * za;
  let b2 = sums.sum_sq_b - 2 * zb * sums.sum_b + d * zb * zb;

  let dot = (a_scale * b_scale) * (ab as f32);
  let norm_sq_a = (a_scale * a_scale) * (a2 as f32);
  let norm_sq_b = (b_scale * b_scale) * (b2 as f32);
  (dot, norm_sq_a, norm_sq_b)
}

pub fn l2_sq_qi8(
  a: &[i8],
  a_scale: f32,
  a_zero_point: i8,
  b: &[i8],
  b_scale: f32,
  b_zero_point: i8,
) -> f32 {
  let (dot, norm_sq_a, norm_sq_b) =
    dot_and_norms_qi8(a, a_scale, a_zero_point, b, b_scale, b_zero_point);
  norm_sq_a + norm_sq_b - 2.0 * dot
}

pub fn inner_product_distance_qi8(
  a: &[i8],
  a_scale: f32,
  a_zero_point: i8,
  b: &[i8],
  b_scale: f32,
  b_zero_point: i8,
) -> f32 {
  let (dot, _, _) = dot_and_norms_qi8(a, a_scale, a_zero_point, b, b_scale, b_zero_point);
  1.0 - dot
}

pub fn cosine_distance_qi8(
  a: &[i8],
  a_scale: f32,
  a_zero_point: i8,
  b: &[i8],
  b_scale: f32,
  b_zero_point: i8,
) -> f32 {
  let (dot, norm_sq_a, norm_sq_b) =
    dot_and_norms_qi8(a, a_scale, a_zero_point, b, b_scale, b_zero_point);
  if norm_sq_a == 0.0 || norm_sq_b == 0.0 {
    return 1.0;
  }
  let denom = (norm_sq_a * norm_sq_b).sqrt();
  if denom == 0.0 {
    1.0
  } else {
    1.0 - dot / denom
  }
}

impl Kernel for f32 {
  fn dot(a: &[Self], b: &[Self]) -> f32 {
    assert_eq!(a.len(), b.len());
    dot_f32(a, b)
  }

  fn l2_sq(a: &[Self], b: &[Self]) -> f32 {
    assert_eq!(a.len(), b.len());
    l2_sq_f32(a, b)
  }

  fn dot_and_norms(a: &[Self], b: &[Self]) -> (f32, f32, f32) {
    assert_eq!(a.len(), b.len());
    dot_and_norms_f32(a, b)
  }
}

impl Kernel for bf16 {
  fn dot(a: &[Self], b: &[Self]) -> f32 {
    assert_eq!(a.len(), b.len());
    dot_bf16(a, b)
  }

  fn l2_sq(a: &[Self], b: &[Self]) -> f32 {
    assert_eq!(a.len(), b.len());
    l2_sq_bf16(a, b)
  }

  fn dot_and_norms(a: &[Self], b: &[Self]) -> (f32, f32, f32) {
    assert_eq!(a.len(), b.len());
    dot_and_norms_bf16(a, b)
  }
}

impl Kernel for f16 {
  fn dot(a: &[Self], b: &[Self]) -> f32 {
    assert_eq!(a.len(), b.len());
    dot_f16(a, b)
  }

  fn l2_sq(a: &[Self], b: &[Self]) -> f32 {
    assert_eq!(a.len(), b.len());
    l2_sq_f16(a, b)
  }

  fn dot_and_norms(a: &[Self], b: &[Self]) -> (f32, f32, f32) {
    assert_eq!(a.len(), b.len());
    dot_and_norms_f16(a, b)
  }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") {
      // SAFETY: guarded by runtime feature check.
      unsafe { return x86::dot_f32_avx512(a, b) };
    }
    if std::is_x86_feature_detected!("avx") {
      unsafe { return x86::dot_f32_avx(a, b) };
    }
    if std::is_x86_feature_detected!("sse") {
      unsafe { return x86::dot_f32_sse(a, b) };
    }
  }
  dot_f32_scalar(a, b)
}

fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") {
      unsafe { return x86::l2_sq_f32_avx512(a, b) };
    }
    if std::is_x86_feature_detected!("avx") {
      unsafe { return x86::l2_sq_f32_avx(a, b) };
    }
    if std::is_x86_feature_detected!("sse") {
      unsafe { return x86::l2_sq_f32_sse(a, b) };
    }
  }
  l2_sq_f32_scalar(a, b)
}

fn dot_and_norms_f32(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") {
      unsafe { return x86::dot_and_norms_f32_avx512(a, b) };
    }
    if std::is_x86_feature_detected!("avx") {
      unsafe { return x86::dot_and_norms_f32_avx(a, b) };
    }
    if std::is_x86_feature_detected!("sse") {
      unsafe { return x86::dot_and_norms_f32_sse(a, b) };
    }
  }
  dot_and_norms_f32_scalar(a, b)
}

fn dot_bf16(a: &[bf16], b: &[bf16]) -> f32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16") {
      unsafe { return x86::dot_bf16_avx512bf16(a, b) };
    }
  }
  dot_bf16_scalar(a, b)
}

fn l2_sq_bf16(a: &[bf16], b: &[bf16]) -> f32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16") {
      unsafe { return x86::l2_sq_bf16_avx512bf16(a, b) };
    }
  }
  l2_sq_bf16_scalar(a, b)
}

fn dot_and_norms_bf16(a: &[bf16], b: &[bf16]) -> (f32, f32, f32) {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16") {
      unsafe { return x86::dot_and_norms_bf16_avx512bf16(a, b) };
    }
  }
  dot_and_norms_bf16_scalar(a, b)
}

fn dot_f16(a: &[f16], b: &[f16]) -> f32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("f16c") {
      unsafe { return x86::dot_f16_avx512f_f16c(a, b) };
    }
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("f16c") {
      unsafe { return x86::dot_f16_avx_f16c(a, b) };
    }
  }
  dot_f16_scalar(a, b)
}

fn l2_sq_f16(a: &[f16], b: &[f16]) -> f32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("f16c") {
      unsafe { return x86::l2_sq_f16_avx512f_f16c(a, b) };
    }
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("f16c") {
      unsafe { return x86::l2_sq_f16_avx_f16c(a, b) };
    }
  }
  l2_sq_f16_scalar(a, b)
}

fn dot_and_norms_f16(a: &[f16], b: &[f16]) -> (f32, f32, f32) {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("f16c") {
      unsafe { return x86::dot_and_norms_f16_avx512f_f16c(a, b) };
    }
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("f16c") {
      unsafe { return x86::dot_and_norms_f16_avx_f16c(a, b) };
    }
  }
  dot_and_norms_f16_scalar(a, b)
}

fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
  a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn l2_sq_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
  a.iter()
    .zip(b)
    .map(|(a, b)| {
      let d = a - b;
      d * d
    })
    .sum()
}

fn dot_and_norms_f32_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
  let mut dot = 0.0f32;
  let mut norm_a = 0.0f32;
  let mut norm_b = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    dot += a * b;
    norm_a += a * a;
    norm_b += b * b;
  }
  (dot, norm_a, norm_b)
}

fn dot_bf16_scalar(a: &[bf16], b: &[bf16]) -> f32 {
  let mut dot = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    dot += a.to_f32() * b.to_f32();
  }
  dot
}

fn l2_sq_bf16_scalar(a: &[bf16], b: &[bf16]) -> f32 {
  let mut sum = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    let d = a.to_f32() - b.to_f32();
    sum += d * d;
  }
  sum
}

fn dot_and_norms_bf16_scalar(a: &[bf16], b: &[bf16]) -> (f32, f32, f32) {
  let mut dot = 0.0f32;
  let mut norm_a = 0.0f32;
  let mut norm_b = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    let a = a.to_f32();
    let b = b.to_f32();
    dot += a * b;
    norm_a += a * a;
    norm_b += b * b;
  }
  (dot, norm_a, norm_b)
}

fn dot_f16_scalar(a: &[f16], b: &[f16]) -> f32 {
  let mut dot = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    dot += a.to_f32() * b.to_f32();
  }
  dot
}

fn l2_sq_f16_scalar(a: &[f16], b: &[f16]) -> f32 {
  let mut sum = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    let d = a.to_f32() - b.to_f32();
    sum += d * d;
  }
  sum
}

fn dot_and_norms_f16_scalar(a: &[f16], b: &[f16]) -> (f32, f32, f32) {
  let mut dot = 0.0f32;
  let mut norm_a = 0.0f32;
  let mut norm_b = 0.0f32;
  for (a, b) in a.iter().zip(b) {
    let a = a.to_f32();
    let b = b.to_f32();
    dot += a * b;
    norm_a += a * a;
    norm_b += b * b;
  }
  (dot, norm_a, norm_b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
  use super::I8Sums;
  use half::bf16;
  use half::f16;
  #[cfg(target_arch = "x86")]
  use std::arch::x86::*;
  #[cfg(target_arch = "x86_64")]
  use std::arch::x86_64::*;

  #[inline]
  pub unsafe fn dot_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= a.len() {
      let va = _mm_loadu_ps(a.as_ptr().add(i));
      let vb = _mm_loadu_ps(b.as_ptr().add(i));
      sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
      i += 4;
    }
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut dot = tmp.iter().sum::<f32>();
    while i < a.len() {
      dot += *a.get_unchecked(i) * *b.get_unchecked(i);
      i += 1;
    }
    dot
  }

  #[inline]
  pub unsafe fn dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= a.len() {
      let va = _mm256_loadu_ps(a.as_ptr().add(i));
      let vb = _mm256_loadu_ps(b.as_ptr().add(i));
      sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
      i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut dot = tmp.iter().sum::<f32>();
    while i < a.len() {
      dot += *a.get_unchecked(i) * *b.get_unchecked(i);
      i += 1;
    }
    dot
  }

  #[inline]
  pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= a.len() {
      let va = _mm512_loadu_ps(a.as_ptr().add(i));
      let vb = _mm512_loadu_ps(b.as_ptr().add(i));
      sum = _mm512_fmadd_ps(va, vb, sum);
      i += 16;
    }
    let mut dot = _mm512_reduce_add_ps(sum);
    while i < a.len() {
      dot += *a.get_unchecked(i) * *b.get_unchecked(i);
      i += 1;
    }
    dot
  }

  #[inline]
  pub unsafe fn l2_sq_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= a.len() {
      let va = _mm_loadu_ps(a.as_ptr().add(i));
      let vb = _mm_loadu_ps(b.as_ptr().add(i));
      let diff = _mm_sub_ps(va, vb);
      sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
      i += 4;
    }
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut l2 = tmp.iter().sum::<f32>();
    while i < a.len() {
      let d = *a.get_unchecked(i) - *b.get_unchecked(i);
      l2 += d * d;
      i += 1;
    }
    l2
  }

  #[inline]
  pub unsafe fn l2_sq_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= a.len() {
      let va = _mm256_loadu_ps(a.as_ptr().add(i));
      let vb = _mm256_loadu_ps(b.as_ptr().add(i));
      let diff = _mm256_sub_ps(va, vb);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
      i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut l2 = tmp.iter().sum::<f32>();
    while i < a.len() {
      let d = *a.get_unchecked(i) - *b.get_unchecked(i);
      l2 += d * d;
      i += 1;
    }
    l2
  }

  #[inline]
  pub unsafe fn l2_sq_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= a.len() {
      let va = _mm512_loadu_ps(a.as_ptr().add(i));
      let vb = _mm512_loadu_ps(b.as_ptr().add(i));
      let diff = _mm512_sub_ps(va, vb);
      sum = _mm512_fmadd_ps(diff, diff, sum);
      i += 16;
    }
    let mut l2 = _mm512_reduce_add_ps(sum);
    while i < a.len() {
      let d = *a.get_unchecked(i) - *b.get_unchecked(i);
      l2 += d * d;
      i += 1;
    }
    l2
  }

  #[inline]
  pub unsafe fn i8_sums_avx2(a: &[i8], b: &[i8]) -> I8Sums {
    debug_assert_eq!(a.len(), b.len());

    let ones_i16 = _mm256_set1_epi16(1);
    let mut dot_acc = _mm256_setzero_si256();
    let mut sum_a_acc = _mm256_setzero_si256();
    let mut sum_b_acc = _mm256_setzero_si256();
    let mut sum_sq_a_acc = _mm256_setzero_si256();
    let mut sum_sq_b_acc = _mm256_setzero_si256();

    let mut i = 0usize;
    while i + 32 <= a.len() {
      let a_i8 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
      let b_i8 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

      let a_lo_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_i8));
      let a_hi_i16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8, 1));
      let b_lo_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8));
      let b_hi_i16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8, 1));

      let prod_lo = _mm256_mullo_epi16(a_lo_i16, b_lo_i16);
      let prod_hi = _mm256_mullo_epi16(a_hi_i16, b_hi_i16);
      dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(prod_lo, ones_i16));
      dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(prod_hi, ones_i16));

      sum_a_acc = _mm256_add_epi32(sum_a_acc, _mm256_madd_epi16(a_lo_i16, ones_i16));
      sum_a_acc = _mm256_add_epi32(sum_a_acc, _mm256_madd_epi16(a_hi_i16, ones_i16));
      sum_b_acc = _mm256_add_epi32(sum_b_acc, _mm256_madd_epi16(b_lo_i16, ones_i16));
      sum_b_acc = _mm256_add_epi32(sum_b_acc, _mm256_madd_epi16(b_hi_i16, ones_i16));

      let a2_lo = _mm256_mullo_epi16(a_lo_i16, a_lo_i16);
      let a2_hi = _mm256_mullo_epi16(a_hi_i16, a_hi_i16);
      let b2_lo = _mm256_mullo_epi16(b_lo_i16, b_lo_i16);
      let b2_hi = _mm256_mullo_epi16(b_hi_i16, b_hi_i16);
      sum_sq_a_acc = _mm256_add_epi32(sum_sq_a_acc, _mm256_madd_epi16(a2_lo, ones_i16));
      sum_sq_a_acc = _mm256_add_epi32(sum_sq_a_acc, _mm256_madd_epi16(a2_hi, ones_i16));
      sum_sq_b_acc = _mm256_add_epi32(sum_sq_b_acc, _mm256_madd_epi16(b2_lo, ones_i16));
      sum_sq_b_acc = _mm256_add_epi32(sum_sq_b_acc, _mm256_madd_epi16(b2_hi, ones_i16));

      i += 32;
    }

    let mut dot_buf = [0i32; 8];
    let mut sum_a_buf = [0i32; 8];
    let mut sum_b_buf = [0i32; 8];
    let mut sum_sq_a_buf = [0i32; 8];
    let mut sum_sq_b_buf = [0i32; 8];
    _mm256_storeu_si256(dot_buf.as_mut_ptr() as *mut __m256i, dot_acc);
    _mm256_storeu_si256(sum_a_buf.as_mut_ptr() as *mut __m256i, sum_a_acc);
    _mm256_storeu_si256(sum_b_buf.as_mut_ptr() as *mut __m256i, sum_b_acc);
    _mm256_storeu_si256(sum_sq_a_buf.as_mut_ptr() as *mut __m256i, sum_sq_a_acc);
    _mm256_storeu_si256(sum_sq_b_buf.as_mut_ptr() as *mut __m256i, sum_sq_b_acc);

    let mut dot = dot_buf.iter().sum::<i32>();
    let mut sum_a = sum_a_buf.iter().sum::<i32>();
    let mut sum_b = sum_b_buf.iter().sum::<i32>();
    let mut sum_sq_a = sum_sq_a_buf.iter().sum::<i32>();
    let mut sum_sq_b = sum_sq_b_buf.iter().sum::<i32>();

    while i < a.len() {
      let av = *a.get_unchecked(i) as i32;
      let bv = *b.get_unchecked(i) as i32;
      dot += av * bv;
      sum_a += av;
      sum_b += bv;
      sum_sq_a += av * av;
      sum_sq_b += bv * bv;
      i += 1;
    }

    I8Sums {
      dot,
      sum_a,
      sum_b,
      sum_sq_a,
      sum_sq_b,
    }
  }

  #[inline]
  pub unsafe fn dot_and_norms_f32_sse(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot_acc = _mm_setzero_ps();
    let mut na_acc = _mm_setzero_ps();
    let mut nb_acc = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= a.len() {
      let va = _mm_loadu_ps(a.as_ptr().add(i));
      let vb = _mm_loadu_ps(b.as_ptr().add(i));
      dot_acc = _mm_add_ps(dot_acc, _mm_mul_ps(va, vb));
      na_acc = _mm_add_ps(na_acc, _mm_mul_ps(va, va));
      nb_acc = _mm_add_ps(nb_acc, _mm_mul_ps(vb, vb));
      i += 4;
    }
    let mut dot_tmp = [0.0f32; 4];
    let mut na_tmp = [0.0f32; 4];
    let mut nb_tmp = [0.0f32; 4];
    _mm_storeu_ps(dot_tmp.as_mut_ptr(), dot_acc);
    _mm_storeu_ps(na_tmp.as_mut_ptr(), na_acc);
    _mm_storeu_ps(nb_tmp.as_mut_ptr(), nb_acc);
    let mut dot = dot_tmp.iter().sum::<f32>();
    let mut na = na_tmp.iter().sum::<f32>();
    let mut nb = nb_tmp.iter().sum::<f32>();
    while i < a.len() {
      let va = *a.get_unchecked(i);
      let vb = *b.get_unchecked(i);
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
      i += 1;
    }
    (dot, na, nb)
  }

  #[inline]
  pub unsafe fn dot_and_norms_f32_avx(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot_acc = _mm256_setzero_ps();
    let mut na_acc = _mm256_setzero_ps();
    let mut nb_acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= a.len() {
      let va = _mm256_loadu_ps(a.as_ptr().add(i));
      let vb = _mm256_loadu_ps(b.as_ptr().add(i));
      dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(va, vb));
      na_acc = _mm256_add_ps(na_acc, _mm256_mul_ps(va, va));
      nb_acc = _mm256_add_ps(nb_acc, _mm256_mul_ps(vb, vb));
      i += 8;
    }
    let mut dot_tmp = [0.0f32; 8];
    let mut na_tmp = [0.0f32; 8];
    let mut nb_tmp = [0.0f32; 8];
    _mm256_storeu_ps(dot_tmp.as_mut_ptr(), dot_acc);
    _mm256_storeu_ps(na_tmp.as_mut_ptr(), na_acc);
    _mm256_storeu_ps(nb_tmp.as_mut_ptr(), nb_acc);
    let mut dot = dot_tmp.iter().sum::<f32>();
    let mut na = na_tmp.iter().sum::<f32>();
    let mut nb = nb_tmp.iter().sum::<f32>();
    while i < a.len() {
      let va = *a.get_unchecked(i);
      let vb = *b.get_unchecked(i);
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
      i += 1;
    }
    (dot, na, nb)
  }

  #[inline]
  pub unsafe fn dot_and_norms_f32_avx512(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot_acc = _mm512_setzero_ps();
    let mut na_acc = _mm512_setzero_ps();
    let mut nb_acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= a.len() {
      let va = _mm512_loadu_ps(a.as_ptr().add(i));
      let vb = _mm512_loadu_ps(b.as_ptr().add(i));
      dot_acc = _mm512_fmadd_ps(va, vb, dot_acc);
      na_acc = _mm512_fmadd_ps(va, va, na_acc);
      nb_acc = _mm512_fmadd_ps(vb, vb, nb_acc);
      i += 16;
    }
    let mut dot = _mm512_reduce_add_ps(dot_acc);
    let mut na = _mm512_reduce_add_ps(na_acc);
    let mut nb = _mm512_reduce_add_ps(nb_acc);
    while i < a.len() {
      let va = *a.get_unchecked(i);
      let vb = *b.get_unchecked(i);
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
      i += 1;
    }
    (dot, na, nb)
  }

  #[inline]
  pub unsafe fn dot_bf16_avx512bf16(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= len {
      let a_ptr = a.as_ptr().add(i) as *const i8;
      let b_ptr = b.as_ptr().add(i) as *const i8;
      let a_i = _mm512_loadu_si512(a_ptr as *const _);
      let b_i = _mm512_loadu_si512(b_ptr as *const _);
      let a_bh: __m512bh = std::mem::transmute(a_i);
      let b_bh: __m512bh = std::mem::transmute(b_i);
      acc = _mm512_dpbf16_ps(acc, a_bh, b_bh);
      i += 32;
    }
    let mut dot = _mm512_reduce_add_ps(acc);
    while i < len {
      dot += (*a.get_unchecked(i)).to_f32() * (*b.get_unchecked(i)).to_f32();
      i += 1;
    }
    dot
  }

  #[inline]
  pub unsafe fn dot_and_norms_bf16_avx512bf16(a: &[bf16], b: &[bf16]) -> (f32, f32, f32) {
    let len = a.len();
    let mut dot_acc = _mm512_setzero_ps();
    let mut na_acc = _mm512_setzero_ps();
    let mut nb_acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= len {
      let a_ptr = a.as_ptr().add(i) as *const i8;
      let b_ptr = b.as_ptr().add(i) as *const i8;
      let a_i = _mm512_loadu_si512(a_ptr as *const _);
      let b_i = _mm512_loadu_si512(b_ptr as *const _);
      let a_bh: __m512bh = std::mem::transmute(a_i);
      let b_bh: __m512bh = std::mem::transmute(b_i);
      dot_acc = _mm512_dpbf16_ps(dot_acc, a_bh, b_bh);
      na_acc = _mm512_dpbf16_ps(na_acc, a_bh, a_bh);
      nb_acc = _mm512_dpbf16_ps(nb_acc, b_bh, b_bh);
      i += 32;
    }
    let mut dot = _mm512_reduce_add_ps(dot_acc);
    let mut na = _mm512_reduce_add_ps(na_acc);
    let mut nb = _mm512_reduce_add_ps(nb_acc);
    while i < len {
      let va = (*a.get_unchecked(i)).to_f32();
      let vb = (*b.get_unchecked(i)).to_f32();
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
      i += 1;
    }
    (dot, na, nb)
  }

  #[inline]
  pub unsafe fn l2_sq_bf16_avx512bf16(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= len {
      let a_ptr = a.as_ptr().add(i) as *const i8;
      let b_ptr = b.as_ptr().add(i) as *const i8;
      let a_i = _mm512_loadu_si512(a_ptr as *const _);
      let b_i = _mm512_loadu_si512(b_ptr as *const _);

      let a_low = _mm512_castsi512_si256(a_i);
      let b_low = _mm512_castsi512_si256(b_i);
      let a_low_ps = _mm512_cvtpbh_ps(std::mem::transmute(a_low));
      let b_low_ps = _mm512_cvtpbh_ps(std::mem::transmute(b_low));

      let a_high = _mm512_extracti64x4_epi64(a_i, 1);
      let b_high = _mm512_extracti64x4_epi64(b_i, 1);
      let a_high_ps = _mm512_cvtpbh_ps(std::mem::transmute(a_high));
      let b_high_ps = _mm512_cvtpbh_ps(std::mem::transmute(b_high));

      let diff_low = _mm512_sub_ps(a_low_ps, b_low_ps);
      let diff_high = _mm512_sub_ps(a_high_ps, b_high_ps);
      acc = _mm512_fmadd_ps(diff_low, diff_low, acc);
      acc = _mm512_fmadd_ps(diff_high, diff_high, acc);
      i += 32;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    while i < len {
      let d = (*a.get_unchecked(i)).to_f32() - (*b.get_unchecked(i)).to_f32();
      sum += d * d;
      i += 1;
    }
    sum
  }

  #[inline]
  pub unsafe fn dot_f16_avx512f_f16c(a: &[f16], b: &[f16]) -> f32 {
    let len = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= len {
      let a_i = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
      let b_i = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
      let a_ps = _mm512_cvtph_ps(a_i);
      let b_ps = _mm512_cvtph_ps(b_i);
      acc = _mm512_fmadd_ps(a_ps, b_ps, acc);
      i += 16;
    }
    let mut dot = _mm512_reduce_add_ps(acc);
    while i < len {
      dot += (*a.get_unchecked(i)).to_f32() * (*b.get_unchecked(i)).to_f32();
      i += 1;
    }
    dot
  }

  #[inline]
  pub unsafe fn l2_sq_f16_avx512f_f16c(a: &[f16], b: &[f16]) -> f32 {
    let len = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= len {
      let a_i = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
      let b_i = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
      let a_ps = _mm512_cvtph_ps(a_i);
      let b_ps = _mm512_cvtph_ps(b_i);
      let diff = _mm512_sub_ps(a_ps, b_ps);
      acc = _mm512_fmadd_ps(diff, diff, acc);
      i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    while i < len {
      let d = (*a.get_unchecked(i)).to_f32() - (*b.get_unchecked(i)).to_f32();
      sum += d * d;
      i += 1;
    }
    sum
  }

  #[inline]
  pub unsafe fn dot_and_norms_f16_avx512f_f16c(a: &[f16], b: &[f16]) -> (f32, f32, f32) {
    let len = a.len();
    let mut dot_acc = _mm512_setzero_ps();
    let mut na_acc = _mm512_setzero_ps();
    let mut nb_acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= len {
      let a_i = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
      let b_i = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
      let a_ps = _mm512_cvtph_ps(a_i);
      let b_ps = _mm512_cvtph_ps(b_i);
      dot_acc = _mm512_fmadd_ps(a_ps, b_ps, dot_acc);
      na_acc = _mm512_fmadd_ps(a_ps, a_ps, na_acc);
      nb_acc = _mm512_fmadd_ps(b_ps, b_ps, nb_acc);
      i += 16;
    }
    let mut dot = _mm512_reduce_add_ps(dot_acc);
    let mut na = _mm512_reduce_add_ps(na_acc);
    let mut nb = _mm512_reduce_add_ps(nb_acc);
    while i < len {
      let va = (*a.get_unchecked(i)).to_f32();
      let vb = (*b.get_unchecked(i)).to_f32();
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
      i += 1;
    }
    (dot, na, nb)
  }

  #[inline]
  pub unsafe fn dot_f16_avx_f16c(a: &[f16], b: &[f16]) -> f32 {
    let len = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
      let a_i = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
      let b_i = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
      let a_ps = _mm256_cvtph_ps(a_i);
      let b_ps = _mm256_cvtph_ps(b_i);
      acc = _mm256_add_ps(acc, _mm256_mul_ps(a_ps, b_ps));
      i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut dot = tmp.iter().sum::<f32>();
    while i < len {
      dot += (*a.get_unchecked(i)).to_f32() * (*b.get_unchecked(i)).to_f32();
      i += 1;
    }
    dot
  }

  #[inline]
  pub unsafe fn l2_sq_f16_avx_f16c(a: &[f16], b: &[f16]) -> f32 {
    let len = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
      let a_i = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
      let b_i = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
      let a_ps = _mm256_cvtph_ps(a_i);
      let b_ps = _mm256_cvtph_ps(b_i);
      let diff = _mm256_sub_ps(a_ps, b_ps);
      acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
      i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
      let d = (*a.get_unchecked(i)).to_f32() - (*b.get_unchecked(i)).to_f32();
      sum += d * d;
      i += 1;
    }
    sum
  }

  #[inline]
  pub unsafe fn dot_and_norms_f16_avx_f16c(a: &[f16], b: &[f16]) -> (f32, f32, f32) {
    let len = a.len();
    let mut dot_acc = _mm256_setzero_ps();
    let mut na_acc = _mm256_setzero_ps();
    let mut nb_acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
      let a_i = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
      let b_i = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
      let a_ps = _mm256_cvtph_ps(a_i);
      let b_ps = _mm256_cvtph_ps(b_i);
      dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(a_ps, b_ps));
      na_acc = _mm256_add_ps(na_acc, _mm256_mul_ps(a_ps, a_ps));
      nb_acc = _mm256_add_ps(nb_acc, _mm256_mul_ps(b_ps, b_ps));
      i += 8;
    }
    let mut dot_tmp = [0.0f32; 8];
    let mut na_tmp = [0.0f32; 8];
    let mut nb_tmp = [0.0f32; 8];
    _mm256_storeu_ps(dot_tmp.as_mut_ptr(), dot_acc);
    _mm256_storeu_ps(na_tmp.as_mut_ptr(), na_acc);
    _mm256_storeu_ps(nb_tmp.as_mut_ptr(), nb_acc);
    let mut dot = dot_tmp.iter().sum::<f32>();
    let mut na = na_tmp.iter().sum::<f32>();
    let mut nb = nb_tmp.iter().sum::<f32>();
    while i < len {
      let va = (*a.get_unchecked(i)).to_f32();
      let vb = (*b.get_unchecked(i)).to_f32();
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
      i += 1;
    }
    (dot, na, nb)
  }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
  use super::I8Sums;
  use std::arch::aarch64::*;

  #[inline]
  pub unsafe fn i8_sums_neon(a: &[i8], b: &[i8]) -> I8Sums {
    debug_assert_eq!(a.len(), b.len());

    let mut dot_acc = vdupq_n_s32(0);
    let mut sum_a_acc = vdupq_n_s32(0);
    let mut sum_b_acc = vdupq_n_s32(0);
    let mut sum_sq_a_acc = vdupq_n_s32(0);
    let mut sum_sq_b_acc = vdupq_n_s32(0);

    let mut i = 0usize;
    while i + 16 <= a.len() {
      let a_i8 = vld1q_s8(a.as_ptr().add(i));
      let b_i8 = vld1q_s8(b.as_ptr().add(i));

      // dot
      let prod_lo = vmull_s8(vget_low_s8(a_i8), vget_low_s8(b_i8));
      let prod_hi = vmull_s8(vget_high_s8(a_i8), vget_high_s8(b_i8));
      dot_acc = vaddq_s32(dot_acc, vpaddlq_s16(prod_lo));
      dot_acc = vaddq_s32(dot_acc, vpaddlq_s16(prod_hi));

      // sum
      let a_lo_i16 = vmovl_s8(vget_low_s8(a_i8));
      let a_hi_i16 = vmovl_s8(vget_high_s8(a_i8));
      let b_lo_i16 = vmovl_s8(vget_low_s8(b_i8));
      let b_hi_i16 = vmovl_s8(vget_high_s8(b_i8));
      sum_a_acc = vaddq_s32(sum_a_acc, vpaddlq_s16(a_lo_i16));
      sum_a_acc = vaddq_s32(sum_a_acc, vpaddlq_s16(a_hi_i16));
      sum_b_acc = vaddq_s32(sum_b_acc, vpaddlq_s16(b_lo_i16));
      sum_b_acc = vaddq_s32(sum_b_acc, vpaddlq_s16(b_hi_i16));

      // sum_sq
      let a2_lo = vmull_s8(vget_low_s8(a_i8), vget_low_s8(a_i8));
      let a2_hi = vmull_s8(vget_high_s8(a_i8), vget_high_s8(a_i8));
      let b2_lo = vmull_s8(vget_low_s8(b_i8), vget_low_s8(b_i8));
      let b2_hi = vmull_s8(vget_high_s8(b_i8), vget_high_s8(b_i8));
      sum_sq_a_acc = vaddq_s32(sum_sq_a_acc, vpaddlq_s16(a2_lo));
      sum_sq_a_acc = vaddq_s32(sum_sq_a_acc, vpaddlq_s16(a2_hi));
      sum_sq_b_acc = vaddq_s32(sum_sq_b_acc, vpaddlq_s16(b2_lo));
      sum_sq_b_acc = vaddq_s32(sum_sq_b_acc, vpaddlq_s16(b2_hi));

      i += 16;
    }

    let mut dot = vaddvq_s32(dot_acc);
    let mut sum_a = vaddvq_s32(sum_a_acc);
    let mut sum_b = vaddvq_s32(sum_b_acc);
    let mut sum_sq_a = vaddvq_s32(sum_sq_a_acc);
    let mut sum_sq_b = vaddvq_s32(sum_sq_b_acc);

    while i < a.len() {
      let av = *a.get_unchecked(i) as i32;
      let bv = *b.get_unchecked(i) as i32;
      dot += av * bv;
      sum_a += av;
      sum_b += bv;
      sum_sq_a += av * av;
      sum_sq_b += bv * bv;
      i += 1;
    }

    I8Sums {
      dot,
      sum_a,
      sum_b,
      sum_sq_a,
      sum_sq_b,
    }
  }
}

#[cfg(test)]
mod qi8_tests {
  use super::*;
  use rand::rngs::StdRng;
  use rand::Rng;
  use rand::SeedableRng;

  fn dequant(v: &[i8], scale: f32, zero: i8) -> Vec<f32> {
    v.iter()
      .map(|&x| (x as i32 - zero as i32) as f32 * scale)
      .collect()
  }

  #[test]
  fn qi8_l2_sq_matches_f32_reference() {
    let mut rng = StdRng::seed_from_u64(123);
    for dim in [1usize, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024] {
      for _ in 0..200 {
        let a = (0..dim).map(|_| rng.gen::<i8>()).collect::<Vec<_>>();
        let b = (0..dim).map(|_| rng.gen::<i8>()).collect::<Vec<_>>();
        let a_scale = rng.gen_range(0.0001..10.0);
        let b_scale = rng.gen_range(0.0001..10.0);
        let a_zero = rng.gen::<i8>();
        let b_zero = rng.gen::<i8>();

        let af = dequant(&a, a_scale, a_zero);
        let bf = dequant(&b, b_scale, b_zero);
        let ref_l2 = l2_sq_f32_scalar(&af, &bf);
        let got = l2_sq_qi8(&a, a_scale, a_zero, &b, b_scale, b_zero);

        let denom = ref_l2.abs().max(1.0);
        assert!(((got - ref_l2) / denom).abs() < 1e-3);
      }
    }
  }
}
