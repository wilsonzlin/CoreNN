use crate::vec::VecData;
use half::bf16;
use half::f16;
use ndarray::ArrayView1;
use ndarray::ArrayView1 as AV;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::mem::transmute;

#[target_feature(enable = "avx512f,avx512bf16")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn dist_l2_bf16_avx512(a_slice: &[bf16], b_slice: &[bf16]) -> f64 {
  let len = a_slice.len();

  let mut acc_sum_ps = _mm512_setzero_ps(); // Accumulator for sum of squares (16 f32s)

  let ptr_a = a_slice.as_ptr();
  let ptr_b = b_slice.as_ptr();

  let mut i = 0;
  // Process chunks of 32 bf16 elements
  let limit_avx512 = len - (len % 32);

  while i < limit_avx512 {
    // Load 32 bf16s from a and b (as __m512i)
    let v_a_i = _mm512_loadu_si512(ptr_a.add(i) as *const _);
    let v_b_i = _mm512_loadu_si512(ptr_b.add(i) as *const _);

    // Convert lower 16 bf16s (from lower 256 bits of __m512i) to f32
    // (AVX-512 doesn't have direct bf16 subtraction)
    let v_a_low_256bh = _mm512_castsi512_si256(v_a_i); // __m256i, represents 16xbf16
    let v_b_low_256bh = _mm512_castsi512_si256(v_b_i);

    let v_a_low_ps = _mm512_cvtpbh_ps(transmute(v_a_low_256bh)); // __m512, 16xf32
    let v_b_low_ps = _mm512_cvtpbh_ps(transmute(v_b_low_256bh));

    // Convert upper 16 bf16s (from upper 256 bits of __m512i) to f32
    let v_a_high_256bh = _mm512_extracti64x4_epi64(v_a_i, 1); // __m256i, represents upper 16xbf16
    let v_b_high_256bh = _mm512_extracti64x4_epi64(v_b_i, 1);

    let v_a_high_ps = _mm512_cvtpbh_ps(transmute(v_a_high_256bh)); // __m512, 16xf32
    let v_b_high_ps = _mm512_cvtpbh_ps(transmute(v_b_high_256bh));

    // Subtract in f32
    let v_diff_low_ps = _mm512_sub_ps(v_a_low_ps, v_b_low_ps);
    let v_diff_high_ps = _mm512_sub_ps(v_a_high_ps, v_b_high_ps);

    // Convert f32 differences (2x16 f32s) back to one __m512bh (32xbf16, represented as __m512i)
    // _mm512_cvtne2ps_pbh(src_high_ps, src_low_ps)
    let v_diff_bh = _mm512_cvtne2ps_pbh(v_diff_high_ps, v_diff_low_ps);

    // Accumulate sum of squares using VDPBF16PS
    // acc_sum_ps = old_acc_sum_ps + dot_product_sum(v_diff_bh, v_diff_bh)
    acc_sum_ps = _mm512_dpbf16_ps(acc_sum_ps, v_diff_bh, v_diff_bh);

    i += 32;
  }

  // Horizontally sum the 16 f32 values in acc_sum_ps
  let mut total_sum_f32 = _mm512_reduce_add_ps(acc_sum_ps);

  // Handle remainder using scalar operations
  if i < len {
    let mut scalar_sum_f32: f32 = 0.0;
    for k in i..len {
      // Bounds are checked by i < len and loop condition k < len
      let val_a_f32 = (*ptr_a.add(k)).to_f32();
      let val_b_f32 = (*ptr_b.add(k)).to_f32();
      let diff_f32 = val_a_f32 - val_b_f32;
      scalar_sum_f32 += diff_f32 * diff_f32;
    }
    total_sum_f32 += scalar_sum_f32;
  }

  // Final sqrt
  let result_f32 = total_sum_f32.sqrt(); // Using f32::sqrt
  result_f32.into()
}

#[target_feature(enable = "avx512f,avx512fp16")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn dist_l2_f16_avx512(a_slice: &[f16], b_slice: &[f16]) -> f64 {
  let len = a_slice.len();
  let mut acc_sum_ps = _mm512_setzero_ps(); // Accumulator for sum of squares (16 f32s)

  let ptr_a = a_slice.as_ptr();
  let ptr_b = b_slice.as_ptr();

  let mut i = 0;
  // Process chunks of 16 f16 elements
  let limit_avx512 = len - (len % 16);

  while i < limit_avx512 {
    // Load 16 f16s from a and b (as __m256i, which represents __m256h)
    // These intrinsics load 256 bits, which is 16 x f16 elements.
    let v_a_ph_i = _mm256_loadu_si256(ptr_a.add(i) as *const _); // __m256i for 16xf16
    let v_b_ph_i = _mm256_loadu_si256(ptr_b.add(i) as *const _); // __m256i for 16xf16

    // Convert 16 f16s to 16 f32s
    let v_a_ps = _mm512_cvtph_ps(v_a_ph_i); // __m512 for 16xf32
    let v_b_ps = _mm512_cvtph_ps(v_b_ph_i);

    // Subtract in f32
    let v_diff_ps = _mm512_sub_ps(v_a_ps, v_b_ps);

    // Square and accumulate: acc_sum_ps = acc_sum_ps + (v_diff_ps * v_diff_ps)
    // Using FMA (fused multiply-add)
    acc_sum_ps = _mm512_fmadd_ps(v_diff_ps, v_diff_ps, acc_sum_ps);

    i += 16;
  }

  // Horizontally sum the 16 f32 values in acc_sum_ps
  let mut total_sum_f32 = _mm512_reduce_add_ps(acc_sum_ps);

  // Handle remainder using scalar operations
  if i < len {
    let mut scalar_sum_f32: f32 = 0.0;
    for k in i..len {
      let val_a_f32 = (*ptr_a.add(k)).to_f32();
      let val_b_f32 = (*ptr_b.add(k)).to_f32();
      let diff_f32 = val_a_f32 - val_b_f32;
      scalar_sum_f32 += diff_f32 * diff_f32;
    }
    total_sum_f32 += scalar_sum_f32;
  }

  let result_f32 = total_sum_f32.sqrt();
  result_f32.into() // Converts f32 to f64
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn dist_l2_f32_avx512(a_slice: &[f32], b_slice: &[f32]) -> f64 {
  let len = a_slice.len();
  let mut acc_sum_ps = _mm512_setzero_ps(); // Accumulator for sum of squares (16 f32s)

  let ptr_a = a_slice.as_ptr();
  let ptr_b = b_slice.as_ptr();

  let mut i = 0;
  // Process chunks of 16 f32 elements
  let limit_avx512 = len - (len % 16);

  while i < limit_avx512 {
    // Load 16 f32s from a and b
    let v_a_ps = _mm512_loadu_ps(ptr_a.add(i));
    let v_b_ps = _mm512_loadu_ps(ptr_b.add(i));

    // Subtract
    let v_diff_ps = _mm512_sub_ps(v_a_ps, v_b_ps);

    // Square and accumulate: acc_sum_ps = acc_sum_ps + (v_diff_ps * v_diff_ps)
    // Using FMA (fused multiply-add)
    acc_sum_ps = _mm512_fmadd_ps(v_diff_ps, v_diff_ps, acc_sum_ps);

    i += 16;
  }

  // Horizontally sum the 16 f32 values in acc_sum_ps
  let mut total_sum_f32 = _mm512_reduce_add_ps(acc_sum_ps);

  // Handle remainder using scalar operations
  if i < len {
    let mut scalar_sum_f32: f32 = 0.0;
    for k in i..len {
      let val_a = *ptr_a.add(k);
      let val_b = *ptr_b.add(k);
      let diff = val_a - val_b;
      scalar_sum_f32 += diff * diff;
    }
    total_sum_f32 += scalar_sum_f32;
  }

  let result_f32 = total_sum_f32.sqrt();
  result_f32.into() // Converts f32 to f64
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn dist_l2_f64_avx512(a_slice: &[f64], b_slice: &[f64]) -> f64 {
  let len = a_slice.len();
  let mut acc_sum_pd = _mm512_setzero_pd(); // Accumulator for sum of squares (8 f64s)

  let ptr_a = a_slice.as_ptr();
  let ptr_b = b_slice.as_ptr();

  let mut i = 0;
  // Process chunks of 8 f64 elements
  let limit_avx512 = len - (len % 8);

  while i < limit_avx512 {
    // Load 8 f64s from a and b
    let v_a_pd = _mm512_loadu_pd(ptr_a.add(i));
    let v_b_pd = _mm512_loadu_pd(ptr_b.add(i));

    // Subtract
    let v_diff_pd = _mm512_sub_pd(v_a_pd, v_b_pd);

    // Square and accumulate: acc_sum_pd = acc_sum_pd + (v_diff_pd * v_diff_pd)
    // Using FMA (fused multiply-add)
    acc_sum_pd = _mm512_fmadd_pd(v_diff_pd, v_diff_pd, acc_sum_pd);

    i += 8;
  }

  // Horizontally sum the 8 f64 values in acc_sum_pd
  let mut total_sum_f64 = _mm512_reduce_add_pd(acc_sum_pd);

  // Handle remainder using scalar operations
  if i < len {
    let mut scalar_sum_f64: f64 = 0.0;
    for k in i..len {
      let val_a = *ptr_a.add(k);
      let val_b = *ptr_b.add(k);
      let diff = val_a - val_b;
      scalar_sum_f64 += diff * diff;
    }
    total_sum_f64 += scalar_sum_f64;
  }

  total_sum_f64.sqrt() // Result is already f64
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve,bf16")]
unsafe fn dist_l2_bf16_sve(a_slice: &[bf16], b_slice: &[bf16]) -> f64 {
  let len = a_slice.len();
  if len == 0 {
    return 0.0;
  }

  let mut p_a = a_slice.as_ptr();
  let mut p_b = b_slice.as_ptr();

  let mut acc_vec_f32 = svdup_n_f32(0.0f32);
  let mut count = len as u64;

  while svptest_first(svptrue_b16(), svwhilelt_b16_u64(0, count)) {
    let pg: svbool_t = svwhilelt_b16_u64(0, count);
    let current_vl = svcnth_pat(SV_ALL) as u64; // Number of active 16-bit elements

    let v_a_bf16 = svld1_bf16(pg, p_a as *const _);
    let v_b_bf16 = svld1_bf16(pg, p_b as *const _);

    let v_diff_bf16 = svsub_bf16_z(pg, v_a_bf16, v_b_bf16);

    acc_vec_f32 = svdot_bf16_f32_z(pg, acc_vec_f32, v_diff_bf16, v_diff_bf16);

    p_a = p_a.add(current_vl as usize);
    p_b = p_b.add(current_vl as usize);
    count -= current_vl;
  }

  let total_sum_f32: f32 = svaddv_f32(svptrue_b32(), acc_vec_f32);
  (total_sum_f32 as f64).sqrt()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve,sve2,fp16")] // "sve2-fp16" for svfmlalb_f32_f16_z
unsafe fn dist_l2_f16_sve(a_slice: &[f16], b_slice: &[f16]) -> f64 {
  let len = a_slice.len();
  if len == 0 {
    return 0.0;
  }

  let mut p_a = a_slice.as_ptr();
  let mut p_b = b_slice.as_ptr();

  let mut acc_vec_f32 = svdup_n_f32(0.0f32);
  let mut count = len as u64;

  while svptest_first(svptrue_b16(), svwhilelt_b16_u64(0, count)) {
    let pg: svbool_t = svwhilelt_b16_u64(0, count);
    let current_vl = svcnth_pat(SV_ALL) as u64;

    let v_a_f16 = svld1_f16(pg, p_a as *const _);
    let v_b_f16 = svld1_f16(pg, p_b as *const _);

    let v_diff_f16 = svsub_f16_z(pg, v_a_f16, v_b_f16);

    // acc_f32 += v_diff_f16 * v_diff_f16 (element-wise, widened)
    acc_vec_f32 = svfmlalb_f32_f16_z(pg, acc_vec_f32, v_diff_f16, v_diff_f16);

    p_a = p_a.add(current_vl as usize);
    p_b = p_b.add(current_vl as usize);
    count -= current_vl;
  }

  let total_sum_f32: f32 = svaddv_f32(svptrue_b32(), acc_vec_f32);
  (total_sum_f32 as f64).sqrt()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn dist_l2_f32_sve(a_slice: &[f32], b_slice: &[f32]) -> f64 {
  let len = a_slice.len();
  if len == 0 {
    return 0.0;
  }

  let mut p_a = a_slice.as_ptr();
  let mut p_b = b_slice.as_ptr();

  let mut acc_vec_f32 = svdup_n_f32(0.0f32);
  let mut count = len as u64;

  while svptest_first(svptrue_b32(), svwhilelt_b32_u64(0, count)) {
    let pg: svbool_t = svwhilelt_b32_u64(0, count);
    let current_vl = svcntw_pat(SV_ALL) as u64; // Number of active 32-bit elements

    let v_a_f32 = svld1_f32(pg, p_a);
    let v_b_f32 = svld1_f32(pg, p_b);

    let v_diff_f32 = svsub_f32_z(pg, v_a_f32, v_b_f32);

    acc_vec_f32 = svfmad_f32_z(pg, acc_vec_f32, v_diff_f32, v_diff_f32);

    p_a = p_a.add(current_vl as usize);
    p_b = p_b.add(current_vl as usize);
    count -= current_vl;
  }

  let total_sum_f32: f32 = svaddv_f32(svptrue_b32(), acc_vec_f32);
  (total_sum_f32 as f64).sqrt()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn dist_l2_f64_sve(a_slice: &[f64], b_slice: &[f64]) -> f64 {
  let len = a_slice.len();
  if len == 0 {
    return 0.0;
  }

  let mut p_a = a_slice.as_ptr();
  let mut p_b = b_slice.as_ptr();

  let mut acc_vec_f64 = svdup_n_f64(0.0f64);
  let mut count = len as u64;

  while svptest_first(svptrue_b64(), svwhilelt_b64_u64(0, count)) {
    let pg: svbool_t = svwhilelt_b64_u64(0, count);
    let current_vl = svcntd_pat(SV_ALL) as u64; // Number of active 64-bit elements

    let v_a_f64 = svld1_f64(pg, p_a);
    let v_b_f64 = svld1_f64(pg, p_b);

    let v_diff_f64 = svsub_f64_z(pg, v_a_f64, v_b_f64);

    acc_vec_f64 = svfmad_f64_z(pg, acc_vec_f64, v_diff_f64, v_diff_f64);

    p_a = p_a.add(current_vl as usize);
    p_b = p_b.add(current_vl as usize);
    count -= current_vl;
  }

  let total_sum_f64: f64 = svaddv_f64(svptrue_b64(), acc_vec_f64);
  total_sum_f64.sqrt()
}

fn dist_l2_scalar<T: num::Float>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
  let diff = a - b;
  let squared_diff = &diff * &diff;
  let sum_squared_diff = squared_diff.sum();
  sum_squared_diff.sqrt().to_f64().unwrap()
}

pub fn dist_l2(a: &VecData, b: &VecData) -> f64 {
  // Avoid unsafe out-of-bounds access if we dispatch to unsafe vectorized implementation.
  assert_eq!(a.dim(), b.dim());

  match (a, b) {
    (VecData::BF16(a_arr), VecData::BF16(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
          unsafe {
            return dist_l2_bf16_avx512(a_arr, b_arr);
          }
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if is_aarch64_feature_detected!("sve") && is_aarch64_feature_detected!("bf16") {
          unsafe {
            return dist_l2_bf16_sve(a_arr, b_arr);
          }
        }
      }
      dist_l2_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    (VecData::F16(a_arr), VecData::F16(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512fp16") {
          unsafe {
            return dist_l2_f16_avx512(a_arr, b_arr);
          }
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        // svfmlalb_f32_f16_z requires SVE2 and FP16 extensions
        if is_aarch64_feature_detected!("sve")
          && is_aarch64_feature_detected!("sve2")
          && is_aarch64_feature_detected!("fp16")
        {
          unsafe {
            return dist_l2_f16_sve(a_arr, b_arr);
          }
        }
      }
      dist_l2_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    (VecData::F32(a_arr), VecData::F32(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") {
          unsafe {
            return dist_l2_f32_avx512(a_arr, b_arr);
          }
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if is_aarch64_feature_detected!("sve") {
          unsafe {
            return dist_l2_f32_sve(a_arr, b_arr);
          }
        }
      }
      dist_l2_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    (VecData::F64(a_arr), VecData::F64(b_arr)) => {
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      {
        if is_x86_feature_detected!("avx512f") {
          unsafe {
            return dist_l2_f64_avx512(a_arr, b_arr);
          }
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if is_aarch64_feature_detected!("sve") {
          unsafe {
            return dist_l2_f64_sve(a_arr, b_arr);
          }
        }
      }
      dist_l2_scalar(&AV::from(a_arr), &AV::from(b_arr))
    }
    _ => panic!("differing dtypes"),
  }
}
