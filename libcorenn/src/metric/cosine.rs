use crate::vec::VecData;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use half::bf16;
use ndarray::ArrayView1;
use ndarray::ArrayView1 as AV;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::mem::transmute;

fn dist_cosine_scalar<T: num::Float + 'static>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let dot_product = a.dot(b);

  let a_norm = a.dot(a);
  let b_norm = b.dot(b);

  let denominator = (a_norm * b_norm).sqrt();

  if denominator == T::zero() {
    T::one()
  } else {
    T::one() - dot_product / denominator
  }
  .to_f64()
  .unwrap()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn dist_cosine_bf16_avx512(a: &[bf16], b: &[bf16]) -> f64 {
  let len = a.len();

  // Accumulators are __m512 (f32) for _mm512_dpbf16_ps
  let mut dot_product_acc_vec = _mm512_setzero_ps();
  let mut a_norm_sq_acc_vec = _mm512_setzero_ps();
  let mut b_norm_sq_acc_vec = _mm512_setzero_ps();

  let mut i = 0;
  let vec_width = 32; // Process 32 bf16 elements at a time

  while i + vec_width <= len {
    let a_ptr = a.as_ptr().add(i) as *const i8;
    let b_ptr = b.as_ptr().add(i) as *const i8;

    let a_vec_i = _mm512_loadu_si512(a_ptr as *const _); // Load 32 u16s
    let b_vec_i = _mm512_loadu_si512(b_ptr as *const _); // Load 32 u16s

    // Transmute __m512i to __m512bh.
    let a_vec_bh: __m512bh = transmute(a_vec_i);
    let b_vec_bh: __m512bh = transmute(b_vec_i);

    dot_product_acc_vec = _mm512_dpbf16_ps(dot_product_acc_vec, a_vec_bh, b_vec_bh);
    a_norm_sq_acc_vec = _mm512_dpbf16_ps(a_norm_sq_acc_vec, a_vec_bh, a_vec_bh);
    b_norm_sq_acc_vec = _mm512_dpbf16_ps(b_norm_sq_acc_vec, b_vec_bh, b_vec_bh);

    i += vec_width;
  }

  // Reduce the __m512 (f32) accumulators
  let mut dot_product_sum = _mm512_reduce_add_ps(dot_product_acc_vec) as f64;
  let mut a_norm_sq_sum = _mm512_reduce_add_ps(a_norm_sq_acc_vec) as f64;
  let mut b_norm_sq_sum = _mm512_reduce_add_ps(b_norm_sq_acc_vec) as f64;

  // Remainder loop
  while i < len {
    let val_a = a[i].to_f64();
    let val_b = b[i].to_f64();
    dot_product_sum += val_a * val_b;
    a_norm_sq_sum += val_a * val_a;
    b_norm_sq_sum += val_b * val_b;
    i += 1;
  }

  if a_norm_sq_sum == 0.0 || b_norm_sq_sum == 0.0 {
    return 1.0;
  }
  let denominator = (a_norm_sq_sum * b_norm_sq_sum).sqrt();
  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product_sum / denominator
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512fp16")]
unsafe fn dist_cosine_f16_avx512(a: &[half::f16], b: &[half::f16]) -> f64 {
  let len = a.len();

  let mut dot_product_sum_vec = _mm512_setzero_ph();
  let mut a_norm_sq_sum_vec = _mm512_setzero_ph();
  let mut b_norm_sq_sum_vec = _mm512_setzero_ph();

  let mut i = 0;
  let vec_width = 32; // 32 f16 elements

  while i + vec_width <= len {
    // half::f16 is repr(transparent) u16, std::f16::f16 is repr(transparent) u16,
    // so it's safe to cast from *half::f16 to *std::f16::f16.
    // This cast is necessary as the intrinsics expect std f16, not half::f16.
    let a_ptr = a.as_ptr().add(i) as *const f16;
    let b_ptr = b.as_ptr().add(i) as *const f16;

    let a_vec = _mm512_loadu_ph(a_ptr);
    let b_vec = _mm512_loadu_ph(b_ptr);

    dot_product_sum_vec = _mm512_fmadd_ph(a_vec, b_vec, dot_product_sum_vec);
    a_norm_sq_sum_vec = _mm512_fmadd_ph(a_vec, a_vec, a_norm_sq_sum_vec);
    b_norm_sq_sum_vec = _mm512_fmadd_ph(b_vec, b_vec, b_norm_sq_sum_vec);

    i += vec_width;
  }

  let dot_sum_f16_scalar: f16 = _mm512_reduce_add_ph(dot_product_sum_vec);
  let a_norm_sq_sum_f16_scalar: f16 = _mm512_reduce_add_ph(a_norm_sq_sum_vec);
  let b_norm_sq_sum_f16_scalar: f16 = _mm512_reduce_add_ph(b_norm_sq_sum_vec);

  let mut dot_product_sum: f64 = dot_sum_f16_scalar.into();
  let mut a_norm_sq_sum: f64 = a_norm_sq_sum_f16_scalar.into();
  let mut b_norm_sq_sum: f64 = b_norm_sq_sum_f16_scalar.into();

  // Remainder loop
  while i < len {
    let val_a: f64 = a[i].into();
    let val_b: f64 = b[i].into();
    dot_product_sum += val_a * val_b;
    a_norm_sq_sum += val_a * val_a;
    b_norm_sq_sum += val_b * val_b;
    i += 1;
  }

  if a_norm_sq_sum == 0.0 || b_norm_sq_sum == 0.0 {
    return 1.0;
  }
  let denominator = (a_norm_sq_sum * b_norm_sq_sum).sqrt();
  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product_sum / denominator
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dist_cosine_f32_avx512(a: &[f32], b: &[f32]) -> f64 {
  let len = a.len();

  let mut dot_product_sum_vec = _mm512_setzero_ps();
  let mut a_norm_sq_sum_vec = _mm512_setzero_ps();
  let mut b_norm_sq_sum_vec = _mm512_setzero_ps();

  let mut i = 0;
  let vec_width = 16; // 16 f32 elements

  while i + vec_width <= len {
    let a_vec = _mm512_loadu_ps(a.as_ptr().add(i) as *const f32);
    let b_vec = _mm512_loadu_ps(b.as_ptr().add(i) as *const f32);

    dot_product_sum_vec = _mm512_fmadd_ps(a_vec, b_vec, dot_product_sum_vec);
    a_norm_sq_sum_vec = _mm512_fmadd_ps(a_vec, a_vec, a_norm_sq_sum_vec);
    b_norm_sq_sum_vec = _mm512_fmadd_ps(b_vec, b_vec, b_norm_sq_sum_vec);

    i += vec_width;
  }

  let mut dot_product_sum = _mm512_reduce_add_ps(dot_product_sum_vec) as f64;
  let mut a_norm_sq_sum = _mm512_reduce_add_ps(a_norm_sq_sum_vec) as f64;
  let mut b_norm_sq_sum = _mm512_reduce_add_ps(b_norm_sq_sum_vec) as f64;

  // Remainder loop
  while i < len {
    let val_a = a[i] as f64;
    let val_b = b[i] as f64;
    dot_product_sum += val_a * val_b;
    a_norm_sq_sum += val_a * val_a;
    b_norm_sq_sum += val_b * val_b;
    i += 1;
  }

  if a_norm_sq_sum == 0.0 || b_norm_sq_sum == 0.0 {
    return 1.0;
  }
  let denominator = (a_norm_sq_sum * b_norm_sq_sum).sqrt();
  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product_sum / denominator
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dist_cosine_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
  let len = a.len();

  let mut dot_product_sum_vec = _mm512_setzero_pd();
  let mut a_norm_sq_sum_vec = _mm512_setzero_pd();
  let mut b_norm_sq_sum_vec = _mm512_setzero_pd();

  let mut i = 0;
  let vec_width = 8; // 8 f64 elements

  while i + vec_width <= len {
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(i) as *const f64);
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(i) as *const f64);

    dot_product_sum_vec = _mm512_fmadd_pd(a_vec, b_vec, dot_product_sum_vec);
    a_norm_sq_sum_vec = _mm512_fmadd_pd(a_vec, a_vec, a_norm_sq_sum_vec);
    b_norm_sq_sum_vec = _mm512_fmadd_pd(b_vec, b_vec, b_norm_sq_sum_vec);

    i += vec_width;
  }

  let mut dot_product_sum = _mm512_reduce_add_pd(dot_product_sum_vec);
  let mut a_norm_sq_sum = _mm512_reduce_add_pd(a_norm_sq_sum_vec);
  let mut b_norm_sq_sum = _mm512_reduce_add_pd(b_norm_sq_sum_vec);

  // Remainder loop
  while i < len {
    let val_a = a[i];
    let val_b = b[i];
    dot_product_sum += val_a * val_b;
    a_norm_sq_sum += val_a * val_a;
    b_norm_sq_sum += val_b * val_b;
    i += 1;
  }

  if a_norm_sq_sum == 0.0 || b_norm_sq_sum == 0.0 {
    return 1.0;
  }
  let denominator = (a_norm_sq_sum * b_norm_sq_sum).sqrt();
  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product_sum / denominator
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dist_cosine_f32_neon(a: &[f32], b: &[f32]) -> f64 {
  let len = a.len();
  if len == 0 {
    return 0.0;
  }

  let mut dot_product_acc = vdupq_n_f32(0.0);
  let mut a_norm_sq_acc = vdupq_n_f32(0.0);
  let mut b_norm_sq_acc = vdupq_n_f32(0.0);

  let mut i = 0;
  let vec_width = 4; // 4 f32 elements per NEON vector

  while i + vec_width <= len {
    let a_vec = vld1q_f32(a.as_ptr().add(i));
    let b_vec = vld1q_f32(b.as_ptr().add(i));

    dot_product_acc = vmlaq_f32(dot_product_acc, a_vec, b_vec);
    a_norm_sq_acc = vmlaq_f32(a_norm_sq_acc, a_vec, a_vec);
    b_norm_sq_acc = vmlaq_f32(b_norm_sq_acc, b_vec, b_vec);

    i += vec_width;
  }

  let mut dot_product_sum = vaddvq_f32(dot_product_acc) as f64;
  let mut a_norm_sq_sum = vaddvq_f32(a_norm_sq_acc) as f64;
  let mut b_norm_sq_sum = vaddvq_f32(b_norm_sq_acc) as f64;

  // Remainder loop
  while i < len {
    let val_a = a[i] as f64;
    let val_b = b[i] as f64;
    dot_product_sum += val_a * val_b;
    a_norm_sq_sum += val_a * val_a;
    b_norm_sq_sum += val_b * val_b;
    i += 1;
  }

  if a_norm_sq_sum == 0.0 || b_norm_sq_sum == 0.0 {
    return 1.0;
  }
  let denominator = (a_norm_sq_sum * b_norm_sq_sum).sqrt();
  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product_sum / denominator
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dist_cosine_f64_neon(a: &[f64], b: &[f64]) -> f64 {
  let len = a.len();
  if len == 0 {
    return 0.0;
  }

  let mut dot_product_acc = vdupq_n_f64(0.0);
  let mut a_norm_sq_acc = vdupq_n_f64(0.0);
  let mut b_norm_sq_acc = vdupq_n_f64(0.0);

  let mut i = 0;
  let vec_width = 2; // 2 f64 elements per NEON vector

  while i + vec_width <= len {
    let a_vec = vld1q_f64(a.as_ptr().add(i));
    let b_vec = vld1q_f64(b.as_ptr().add(i));

    dot_product_acc = vmlaq_f64(dot_product_acc, a_vec, b_vec);
    a_norm_sq_acc = vmlaq_f64(a_norm_sq_acc, a_vec, a_vec);
    b_norm_sq_acc = vmlaq_f64(b_norm_sq_acc, b_vec, b_vec);

    i += vec_width;
  }

  let mut dot_product_sum = vaddvq_f64(dot_product_acc);
  let mut a_norm_sq_sum = vaddvq_f64(a_norm_sq_acc);
  let mut b_norm_sq_sum = vaddvq_f64(b_norm_sq_acc);

  // Remainder loop
  while i < len {
    let val_a = a[i];
    let val_b = b[i];
    dot_product_sum += val_a * val_b;
    a_norm_sq_sum += val_a * val_a;
    b_norm_sq_sum += val_b * val_b;
    i += 1;
  }

  if a_norm_sq_sum == 0.0 || b_norm_sq_sum == 0.0 {
    return 1.0;
  }
  let denominator = (a_norm_sq_sum * b_norm_sq_sum).sqrt();
  if denominator == 0.0 {
    1.0
  } else {
    1.0 - dot_product_sum / denominator
  }
}

pub fn dist_cosine(a: &VecData, b: &VecData) -> f64 {
  // Avoid unsafe out-of-bounds access if we dispatch to unsafe vectorized implementation.
  assert_eq!(a.dim(), b.dim());

  match (a, b) {
    (VecData::BF16(a_arr), VecData::BF16(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
          unsafe {
            return dist_cosine_bf16_avx512(a_arr, b_arr);
          }
        }
      }
      dist_cosine_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    (VecData::F16(a_arr), VecData::F16(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512fp16") {
          unsafe {
            return dist_cosine_f16_avx512(a_arr, b_arr);
          }
        }
      }
      dist_cosine_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    (VecData::F32(a_arr), VecData::F32(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") {
          unsafe {
            return dist_cosine_f32_avx512(a_arr, b_arr);
          }
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if is_aarch64_feature_detected!("neon") {
          unsafe {
            return dist_cosine_f32_neon(a_arr, b_arr);
          }
        }
      }
      dist_cosine_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    (VecData::F64(a_arr), VecData::F64(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") {
          unsafe {
            return dist_cosine_f64_avx512(a_arr, b_arr);
          }
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if is_aarch64_feature_detected!("neon") {
          unsafe {
            return dist_cosine_f64_neon(a_arr, b_arr);
          }
        }
      }
      dist_cosine_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    _ => panic!("differing dtypes"),
  }
}
