//! Scalar Quantization (SQ) Compressor
//! 
//! Scalar quantization maps each float dimension to an 8-bit integer.
//! This provides 4x memory reduction with fast SIMD-friendly distance computation.
//! 
//! The quantization formula is:
//!   q = round((x - min) / (max - min) * 255)
//! 
//! For L2 distance, we can compute in quantized space directly.
//! For cosine, we dequantize and compute (or use lookup tables).

use super::Compressor;
use super::DistanceTable;
use super::CV;
use crate::metric::StdMetric;
use crate::vec::VecData;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;

/// Scalar quantization parameters learned from training data.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScalarQuantizer {
    /// Number of dimensions
    dims: usize,
    /// Minimum value per dimension
    mins: Vec<f32>,
    /// Scale factor per dimension: 255 / (max - min)
    scales: Vec<f32>,
    /// Inverse scale for dequantization: (max - min) / 255
    inv_scales: Vec<f32>,
}

/// Distance lookup table for asymmetric scalar quantization.
/// Precomputes (query[i] - min[i]) * scale[i] for fast distance computation.
#[derive(Debug)]
pub struct SQDistanceTable {
    /// Query values scaled to quantized space: (query - min) * scale
    /// These are f32 to allow fractional values for asymmetric distance.
    scaled_query: Vec<f32>,
    metric: StdMetric,
    /// For cosine: precomputed query norm squared
    query_norm_sq: f32,
}

impl ScalarQuantizer {
    /// Train scalar quantizer from sample vectors.
    /// Computes per-dimension min/max from the training data.
    pub fn train(samples: &[Vec<f32>]) -> Self {
        assert!(!samples.is_empty(), "Need at least one sample");
        let dims = samples[0].len();
        
        // Initialize with first sample
        let mut mins: Vec<f32> = samples[0].clone();
        let mut maxs: Vec<f32> = samples[0].clone();
        
        // Find min/max per dimension
        for sample in samples.iter().skip(1) {
            assert_eq!(sample.len(), dims);
            for (i, &val) in sample.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }
        
        // Compute scales with epsilon to avoid division by zero
        let epsilon = 1e-10;
        let mut scales = Vec::with_capacity(dims);
        let mut inv_scales = Vec::with_capacity(dims);
        
        for i in 0..dims {
            let range = (maxs[i] - mins[i]).max(epsilon);
            scales.push(255.0 / range);
            inv_scales.push(range / 255.0);
        }
        
        ScalarQuantizer {
            dims,
            mins,
            scales,
            inv_scales,
        }
    }
    
    /// Train from CoreNN database by sampling vectors.
    pub fn train_from_corenn(corenn: &crate::CoreNN) -> Self {
        use crate::store::schema::NODE;
        use rand::seq::IteratorRandom;
        
        let sample_size = corenn.cfg.pq_sample_size;
        let mut rng = rand::thread_rng();
        
        // Sample vectors from the database
        let samples: Vec<Vec<f32>> = NODE
            .iter(&corenn.db)
            .choose_multiple(&mut rng, sample_size)
            .into_iter()
            .map(|(_, node)| {
                let vec = node.vector;
                match vec.as_ref() {
                    VecData::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
                    VecData::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
                    VecData::F32(v) => v.clone(),
                    VecData::F64(v) => v.iter().map(|x| *x as f32).collect(),
                }
            })
            .collect();
        
        if samples.is_empty() {
            panic!("Cannot train SQ: no vectors in database");
        }
        
        Self::train(&samples)
    }
    
    /// Quantize a vector to u8 values.
    #[inline]
    pub fn quantize(&self, vec: &[f32]) -> Vec<u8> {
        assert_eq!(vec.len(), self.dims);
        vec.iter()
            .zip(self.mins.iter())
            .zip(self.scales.iter())
            .map(|((&v, &min), &scale)| {
                let q = ((v - min) * scale).round();
                q.clamp(0.0, 255.0) as u8
            })
            .collect()
    }
    
    /// Dequantize u8 values back to f32 (lossy).
    #[inline]
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        quantized.iter()
            .zip(self.mins.iter())
            .zip(self.inv_scales.iter())
            .map(|((&q, &min), &inv_scale)| {
                min + (q as f32) * inv_scale
            })
            .collect()
    }
    
    /// Create distance table for asymmetric distance computation.
    pub fn create_distance_table(&self, query: &[f32], metric: StdMetric) -> SQDistanceTable {
        assert_eq!(query.len(), self.dims);
        
        // Scale query to quantized space (but keep as f32 for precision)
        let scaled_query: Vec<f32> = query.iter()
            .zip(self.mins.iter())
            .zip(self.scales.iter())
            .map(|((&v, &min), &scale)| (v - min) * scale)
            .collect();
        
        let query_norm_sq = if metric == StdMetric::Cosine {
            query.iter().map(|x| x * x).sum()
        } else {
            0.0
        };
        
        SQDistanceTable {
            scaled_query,
            metric,
            query_norm_sq,
        }
    }
    
    /// Compute L2 distance using the distance table.
    /// This is asymmetric: query is not quantized, target is quantized.
    #[inline]
    pub fn distance_l2(&self, table: &SQDistanceTable, quantized: &[u8]) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { self.distance_l2_avx512(table, quantized) };
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { self.distance_l2_neon(table, quantized) };
            }
        }
        
        self.distance_l2_scalar(table, quantized)
    }
    
    /// Scalar fallback for L2 distance.
    #[inline]
    fn distance_l2_scalar(&self, table: &SQDistanceTable, quantized: &[u8]) -> f64 {
        let mut original_sum_sq: f32 = 0.0;
        for (i, &q) in quantized.iter().enumerate() {
            let scaled_diff = table.scaled_query[i] - (q as f32);
            let original_diff = scaled_diff * self.inv_scales[i];
            original_sum_sq += original_diff * original_diff;
        }
        (original_sum_sq as f64).sqrt()
    }
    
    /// AVX-512 optimized L2 distance.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn distance_l2_avx512(&self, table: &SQDistanceTable, quantized: &[u8]) -> f64 {
        use std::arch::x86_64::*;
        
        let n = quantized.len();
        let mut i = 0;
        
        // Process 16 elements at a time with AVX-512
        let mut acc = _mm512_setzero_ps();
        
        while i + 16 <= n {
            // Load 16 u8 values and convert to f32
            let q_bytes = _mm_loadu_si128(quantized.as_ptr().add(i) as *const _);
            let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
            let q_f32 = _mm512_cvtepi32_ps(q_i32);
            
            // Load scaled query and inv_scales
            let sq = _mm512_loadu_ps(table.scaled_query.as_ptr().add(i));
            let inv_s = _mm512_loadu_ps(self.inv_scales.as_ptr().add(i));
            
            // Compute (scaled_query - quantized) * inv_scale
            let diff = _mm512_sub_ps(sq, q_f32);
            let orig_diff = _mm512_mul_ps(diff, inv_s);
            
            // Accumulate squared differences
            acc = _mm512_fmadd_ps(orig_diff, orig_diff, acc);
            
            i += 16;
        }
        
        // Horizontal sum
        let mut sum_sq = _mm512_reduce_add_ps(acc);
        
        // Handle remaining elements
        for j in i..n {
            let scaled_diff = table.scaled_query[j] - (quantized[j] as f32);
            let original_diff = scaled_diff * self.inv_scales[j];
            sum_sq += original_diff * original_diff;
        }
        
        (sum_sq as f64).sqrt()
    }
    
    /// NEON optimized L2 distance for ARM.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn distance_l2_neon(&self, table: &SQDistanceTable, quantized: &[u8]) -> f64 {
        use std::arch::aarch64::*;
        
        let n = quantized.len();
        let mut sum_sq: f32 = 0.0;
        let mut i = 0;
        
        let mut acc = vdupq_n_f32(0.0);
        
        while i + 4 <= n {
            // Load 4 u8 values
            let q_u8 = vld1_lane_u8::<0>(quantized.as_ptr().add(i), vdup_n_u8(0));
            let q_u8 = vld1_lane_u8::<1>(quantized.as_ptr().add(i + 1), q_u8);
            let q_u8 = vld1_lane_u8::<2>(quantized.as_ptr().add(i + 2), q_u8);
            let q_u8 = vld1_lane_u8::<3>(quantized.as_ptr().add(i + 3), q_u8);
            
            // Convert to f32
            let q_u16 = vmovl_u8(q_u8);
            let q_u32 = vmovl_u16(vget_low_u16(q_u16));
            let q_f32 = vcvtq_f32_u32(q_u32);
            
            // Load scaled query and inv_scales
            let sq = vld1q_f32(table.scaled_query.as_ptr().add(i));
            let inv_s = vld1q_f32(self.inv_scales.as_ptr().add(i));
            
            // Compute (scaled_query - quantized) * inv_scale
            let diff = vsubq_f32(sq, q_f32);
            let orig_diff = vmulq_f32(diff, inv_s);
            
            // Accumulate squared differences
            acc = vfmaq_f32(acc, orig_diff, orig_diff);
            
            i += 4;
        }
        
        // Horizontal sum
        sum_sq = vaddvq_f32(acc);
        
        // Handle remaining elements
        for j in i..n {
            let scaled_diff = table.scaled_query[j] - (quantized[j] as f32);
            let original_diff = scaled_diff * self.inv_scales[j];
            sum_sq += original_diff * original_diff;
        }
        
        (sum_sq as f64).sqrt()
    }
    
    /// Compute cosine distance using dequantization.
    #[inline]
    pub fn distance_cosine(&self, table: &SQDistanceTable, quantized: &[u8]) -> f64 {
        // Dequantize and compute cosine
        let dequantized = self.dequantize(quantized);
        
        let mut dot_product: f32 = 0.0;
        let mut target_norm_sq: f32 = 0.0;
        
        // Compute original query values from scaled
        for (i, &q) in dequantized.iter().enumerate() {
            let query_val = table.scaled_query[i] * self.inv_scales[i] + self.mins[i];
            dot_product += query_val * q;
            target_norm_sq += q * q;
        }
        
        let denom = (table.query_norm_sq * target_norm_sq).sqrt();
        if denom < 1e-10 {
            return if table.query_norm_sq < 1e-10 && target_norm_sq < 1e-10 {
                0.0
            } else {
                1.0
            };
        }
        
        let cosine_sim = (dot_product / denom) as f64;
        1.0 - cosine_sim.clamp(-1.0, 1.0)
    }
}

impl Compressor for ScalarQuantizer {
    fn into_compressed(&self, v: VecData) -> CV {
        let v_f32: Vec<f32> = match v {
            VecData::BF16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            VecData::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            VecData::F32(v) => v,
            VecData::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        };
        Arc::new(self.quantize(&v_f32))
    }
    
    fn create_distance_table(&self, query: &VecData, metric: StdMetric) -> Option<DistanceTable> {
        let query_f32: Vec<f32> = match query {
            VecData::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
            VecData::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            VecData::F32(v) => v.clone(),
            VecData::F64(v) => v.iter().map(|x| *x as f32).collect(),
        };
        Some(Arc::new(self.create_distance_table(&query_f32, metric)))
    }
    
    fn dist_with_table(&self, table: &DistanceTable, cv: &CV) -> Option<f64> {
        let table = table.downcast_ref::<SQDistanceTable>()?;
        let quantized = cv.downcast_ref::<Vec<u8>>()?;
        
        Some(match table.metric {
            StdMetric::L2 => self.distance_l2(table, quantized),
            StdMetric::Cosine => self.distance_cosine(table, quantized),
        })
    }
    
    fn dist(&self, metric: StdMetric, a: &CV, b: &CV) -> f64 {
        let a_q = a.downcast_ref::<Vec<u8>>().unwrap();
        let b_q = b.downcast_ref::<Vec<u8>>().unwrap();
        
        // Dequantize and compute distance
        let a_f = self.dequantize(a_q);
        let b_f = self.dequantize(b_q);
        
        match metric {
            StdMetric::L2 => {
                let sum_sq: f32 = a_f.iter()
                    .zip(b_f.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (sum_sq as f64).sqrt()
            }
            StdMetric::Cosine => {
                let dot: f32 = a_f.iter().zip(b_f.iter()).map(|(a, b)| a * b).sum();
                let norm_a: f32 = a_f.iter().map(|x| x * x).sum();
                let norm_b: f32 = b_f.iter().map(|x| x * x).sum();
                let denom = (norm_a * norm_b).sqrt();
                if denom < 1e-10 {
                    1.0
                } else {
                    1.0 - ((dot / denom) as f64).clamp(-1.0, 1.0)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let samples = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![0.5, 1.5, 2.5],
        ];
        let sq = ScalarQuantizer::train(&samples);
        
        let original = vec![0.5, 1.5, 2.5];
        let quantized = sq.quantize(&original);
        let dequantized = sq.dequantize(&quantized);
        
        // Should be close to original
        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.02, "Dequantized value should be close to original");
        }
    }
    
    #[test]
    fn test_distance_ordering() {
        let samples: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();
        let sq = ScalarQuantizer::train(&samples);
        
        let query = vec![50.0, 100.0, 150.0];
        let close = vec![51.0, 102.0, 153.0];
        let far = vec![80.0, 160.0, 240.0];
        
        let table = sq.create_distance_table(&query, StdMetric::L2);
        
        let close_q = sq.quantize(&close);
        let far_q = sq.quantize(&far);
        
        let d_close = sq.distance_l2(&table, &close_q);
        let d_far = sq.distance_l2(&table, &far_q);
        
        assert!(d_close < d_far, "Close should be closer than far: {} vs {}", d_close, d_far);
    }
}
