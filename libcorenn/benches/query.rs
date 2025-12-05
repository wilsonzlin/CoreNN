//! Benchmarks for full query path
//! 
//! Run with: cargo bench -p libcorenn --bench query

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use libcorenn::cfg::Cfg;
use libcorenn::metric::StdMetric;
use libcorenn::CoreNN;
use rand::Rng;

fn random_f32_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn bench_query_throughput(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    
    // Test different dataset sizes
    let sizes = [100, 1000, 10000];
    
    let mut group = c.benchmark_group("query_throughput");
    
    for &n in &sizes {
        // Create in-memory database
        let cfg = Cfg {
            dim,
            metric: StdMetric::L2,
            beam_width: 4,
            max_edges: 32,
            query_search_list_cap: 128,
            update_search_list_cap: 128,
            ..Default::default()
        };
        
        let db = CoreNN::new_in_memory(cfg);
        
        // Insert n vectors
        for i in 0..n {
            let v = random_f32_vec(dim);
            db.insert(&format!("vec_{}", i), &v);
        }
        
        // Generate query
        let query = random_f32_vec(dim);
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| db.query(black_box(&query), k));
        });
    }
    
    group.finish();
}

fn bench_query_scaling(c: &mut Criterion) {
    let dim = 768; // Common embedding dimension
    let n = 5000;
    let k = 10;
    
    // Create database with 5k vectors
    let cfg = Cfg {
        dim,
        metric: StdMetric::L2,
        beam_width: 4,
        max_edges: 32,
        query_search_list_cap: 128,
        update_search_list_cap: 128,
        ..Default::default()
    };
    
    let db = CoreNN::new_in_memory(cfg);
    
    for i in 0..n {
        let v = random_f32_vec(dim);
        db.insert(&format!("vec_{}", i), &v);
    }
    
    let query = random_f32_vec(dim);
    
    let mut group = c.benchmark_group("query_768d_5k");
    
    // Benchmark different k values
    for &k_val in &[1, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(format!("k={}", k_val)), &k_val, |bencher, &k| {
            bencher.iter(|| db.query(black_box(&query), k));
        });
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_query_throughput, bench_query_scaling
);
criterion_main!(benches);
