//! Benchmarks for distance computations
//! 
//! Run with: cargo bench -p libcorenn

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use libcorenn::vec::VecData;
use libcorenn::metric::l2::dist_l2;
use libcorenn::metric::cosine::dist_cosine;
use rand::Rng;

fn random_f32_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
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

criterion_group!(benches, bench_l2_distance, bench_cosine_distance);
criterion_main!(benches);
