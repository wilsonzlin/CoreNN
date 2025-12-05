//! Benchmarks for distance computations
//! 
//! Run with: cargo bench -p libcorenn

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use libcorenn::vec::VecData;
use libcorenn::metric::l2::dist_l2;
use libcorenn::metric::cosine::dist_cosine;
use libcorenn::metric::StdMetric;
use libcorenn::compressor::pq::ProductQuantizer;
use libcorenn::compressor::scalar::ScalarQuantizer;
use libcorenn::compressor::Compressor;
use ndarray::Array2;
use rand::Rng;

fn random_f32_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn random_f32_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen::<f32>())
}

fn bench_l2_distance(c: &mut Criterion) {
    let dims = [128, 384, 768, 1536];
    let mut group = c.benchmark_group("l2_distance");
    
    for dim in dims {
        let a = VecData::F32(random_f32_vec(dim));
        let b = VecData::F32(random_f32_vec(dim));
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| dist_l2(black_box(&a), black_box(&b)));
        });
    }
    
    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let dims = [128, 384, 768, 1536];
    let mut group = c.benchmark_group("cosine_distance");
    
    for dim in dims {
        let a = VecData::F32(random_f32_vec(dim));
        let b = VecData::F32(random_f32_vec(dim));
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| dist_cosine(black_box(&a), black_box(&b)));
        });
    }
    
    group.finish();
}

fn bench_pq_adc_distance(c: &mut Criterion) {
    // Train PQ on sample data
    let dim = 768;
    let subspaces = 64;  // 768 / 64 = 12 dims per subspace
    let n_training = 1000;
    
    let training_data = random_f32_matrix(n_training, dim);
    let pq = ProductQuantizer::<f32>::train(&training_data.view(), subspaces);
    
    // Create query and target
    let query = VecData::F32(random_f32_vec(dim));
    let target = VecData::F32(random_f32_vec(dim));
    let target_cv = pq.into_compressed(target);
    let target_codes = target_cv.downcast_ref::<Vec<u8>>().unwrap();
    
    // Create distance table
    let query_arr = match &query {
        VecData::F32(v) => ndarray::Array1::from(v.clone()),
        _ => panic!("Expected F32"),
    };
    let dist_table = pq.create_distance_table(&query_arr, StdMetric::L2);
    
    let mut group = c.benchmark_group("pq_adc");
    
    // Benchmark ADC distance
    group.bench_function("adc_768d_64sub", |b| {
        b.iter(|| dist_table.distance(black_box(target_codes)));
    });
    
    // Also benchmark symmetric PQ distance for comparison
    let query_cv = pq.into_compressed(query.clone());
    group.bench_function("symmetric_768d_64sub", |b| {
        b.iter(|| pq.dist(StdMetric::L2, black_box(&query_cv), black_box(&target_cv)));
    });
    
    group.finish();
}

fn bench_sq_distance(c: &mut Criterion) {
    // Train SQ on sample data
    let dim = 768;
    let n_training = 1000;
    
    let samples: Vec<Vec<f32>> = (0..n_training)
        .map(|_| random_f32_vec(dim))
        .collect();
    let sq = ScalarQuantizer::train(&samples);
    
    // Create query and target
    let query = random_f32_vec(dim);
    let target = random_f32_vec(dim);
    let target_q = sq.quantize(&target);
    
    // Create distance table
    let dist_table = sq.create_distance_table(&query, StdMetric::L2);
    
    let mut group = c.benchmark_group("sq_distance");
    
    // Benchmark SQ ADC distance
    group.bench_function("sq_adc_768d", |b| {
        b.iter(|| sq.distance_l2(black_box(&dist_table), black_box(&target_q)));
    });
    
    // Benchmark SQ symmetric distance (dequantize and compute)
    let query_q = sq.quantize(&query);
    group.bench_function("sq_dequantize_768d", |b| {
        b.iter(|| {
            let q = sq.dequantize(black_box(&query_q));
            let t = sq.dequantize(black_box(&target_q));
            let mut sum: f32 = 0.0;
            for i in 0..q.len() {
                let d = q[i] - t[i];
                sum += d * d;
            }
            sum.sqrt()
        });
    });
    
    // Benchmark raw f32 L2 for comparison
    let a = VecData::F32(query.clone());
    let b = VecData::F32(target.clone());
    group.bench_function("raw_f32_768d", |b_iter| {
        b_iter.iter(|| dist_l2(black_box(&a), black_box(&b)));
    });
    
    group.finish();
}

criterion_group!(benches, bench_l2_distance, bench_cosine_distance, bench_pq_adc_distance, bench_sq_distance);
criterion_main!(benches);
