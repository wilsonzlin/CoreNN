//! CI Benchmark Binary

use libcorenn::cfg::Cfg;
use libcorenn::cfg::CompressionMode;
use libcorenn::metric::StdMetric;
use libcorenn::CoreNN;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Deserialize;
use serde::Serialize;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

const DATASETS_DIR: &str = env!("CARGO_MANIFEST_DIR");

#[derive(Debug, Deserialize)]
struct DatasetInfo {
  dtype: String,
  metric: String,
  dim: usize,
  n: usize,
  #[serde(default)]
  q: usize,
  #[serde(default)]
  k: usize,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
  name: String,
  dataset: String,
  dimension: usize,
  num_vectors: usize,
  num_queries: usize,
  k: usize,
  metric: String,
  insert_throughput_vps: f64,
  insert_total_ms: f64,
  #[serde(skip_serializing_if = "Option::is_none")]
  query_throughput_qps: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  query_latency_mean_us: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  query_latency_p50_us: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  query_latency_p95_us: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  query_latency_p99_us: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  recall_at_k: Option<f64>,
  config: BenchmarkConfig,
}

#[derive(Debug, Serialize)]
struct BenchmarkConfig {
  beam_width: usize,
  max_edges: usize,
  query_search_list_cap: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  compression: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
  commit: String,
  timestamp: String,
  results: Vec<BenchmarkResult>,
}

fn random_f32_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
  (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn load_f32_vectors(path: &Path, dim: usize, count: usize) -> Vec<Vec<f32>> {
  let file =
    File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e));
  let mut reader = BufReader::new(file);
  let mut buffer = vec![0u8; dim * 4];
  let mut vectors = Vec::with_capacity(count);

  for _ in 0..count {
    if reader.read_exact(&mut buffer).is_err() {
      break;
    }
    let vec: Vec<f32> = buffer
      .chunks(4)
      .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
      .collect();
    vectors.push(vec);
  }

  vectors
}

fn load_f16_vectors(path: &Path, dim: usize, count: usize) -> Vec<Vec<f32>> {
  let file =
    File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e));
  let mut reader = BufReader::new(file);
  let mut buffer = vec![0u8; dim * 2];
  let mut vectors = Vec::with_capacity(count);

  for _ in 0..count {
    if reader.read_exact(&mut buffer).is_err() {
      break;
    }
    let vec: Vec<f32> = buffer
      .chunks(2)
      .map(|b| {
        let bits = u16::from_le_bytes([b[0], b[1]]);
        half::f16::from_bits(bits).to_f32()
      })
      .collect();
    vectors.push(vec);
  }

  vectors
}

fn load_groundtruth(path: &Path, q: usize, k: usize) -> Vec<Vec<u32>> {
  let file =
    File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e));
  let mut reader = BufReader::new(file);
  let mut buffer = vec![0u8; k * 4];
  let mut results = Vec::with_capacity(q);

  for _ in 0..q {
    if reader.read_exact(&mut buffer).is_err() {
      break;
    }
    let ids: Vec<u32> = buffer
      .chunks(4)
      .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
      .collect();
    results.push(ids);
  }

  results
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
  if sorted.is_empty() {
    return 0.0;
  }
  let idx = ((sorted.len() as f64) * p / 100.0).floor() as usize;
  sorted[idx.min(sorted.len() - 1)]
}

fn benchmark_insert(db: &CoreNN, vectors: &[Vec<f32>]) -> (f64, f64) {
  let start = Instant::now();
  for (i, vec) in vectors.iter().enumerate() {
    db.insert(&format!("vec_{}", i), vec);
  }
  let elapsed = start.elapsed();
  let throughput = vectors.len() as f64 / elapsed.as_secs_f64();
  (throughput, elapsed.as_secs_f64() * 1000.0)
}

fn benchmark_queries(
  db: &CoreNN,
  queries: &[Vec<f32>],
  k: usize,
) -> (f64, f64, f64, f64, f64, Vec<Vec<String>>) {
  let mut latencies = Vec::with_capacity(queries.len());
  let mut all_results = Vec::with_capacity(queries.len());

  for query in queries {
    let start = Instant::now();
    let results = db.query(query, k);
    let elapsed = start.elapsed();
    latencies.push(elapsed.as_secs_f64() * 1_000_000.0);
    all_results.push(results.into_iter().map(|(key, _dist)| key).collect());
  }

  latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
  let p50 = percentile(&latencies, 50.0);
  let p95 = percentile(&latencies, 95.0);
  let p99 = percentile(&latencies, 99.0);
  let total_time: f64 = latencies.iter().sum();
  let qps = queries.len() as f64 / (total_time / 1_000_000.0);

  (qps, mean, p50, p95, p99, all_results)
}

fn compute_recall(results: &[Vec<String>], groundtruth: &[Vec<u32>], k: usize) -> f64 {
  let mut total_correct = 0;
  let mut total = 0;

  for (result, gt) in results.iter().zip(groundtruth.iter()) {
    let gt_set: std::collections::HashSet<u32> = gt.iter().take(k).copied().collect();
    for key in result.iter().take(k) {
      if let Some(id_str) = key.strip_prefix("vec_") {
        if let Ok(id) = id_str.parse::<u32>() {
          if gt_set.contains(&id) {
            total_correct += 1;
          }
        }
      }
    }
    total += k.min(gt.len());
  }

  if total == 0 {
    0.0
  } else {
    total_correct as f64 / total as f64
  }
}

fn parse_metric(s: &str) -> StdMetric {
  match s.to_lowercase().as_str() {
    "l2" | "euclidean" => StdMetric::L2,
    "cosine" => StdMetric::Cosine,
    _ => StdMetric::L2,
  }
}

fn run_random_benchmarks() -> Vec<BenchmarkResult> {
  let mut results = Vec::new();
  let mut rng = StdRng::seed_from_u64(42);

  let base_configs = [
    (128, 1000, "random_128d_1k"),
    (128, 10000, "random_128d_10k"),
    (128, 50000, "random_128d_50k"),
    (384, 10000, "random_384d_10k"),
    (768, 10000, "random_768d_10k"),
    (1536, 5000, "random_1536d_5k"),
  ];

  let k = 10;
  let num_queries = 100;

  for (dim, num_vectors, name) in base_configs {
    info!(benchmark = name, "starting");

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
      .map(|_| random_f32_vec(&mut rng, dim))
      .collect();
    let queries: Vec<Vec<f32>> = (0..num_queries)
      .map(|_| random_f32_vec(&mut rng, dim))
      .collect();

    let cfg = Cfg {
      dim,
      metric: StdMetric::L2,
      beam_width: 4,
      max_edges: 32,
      query_search_list_cap: 128,
      update_search_list_cap: 128,
      compression_threshold: usize::MAX,
      ..Default::default()
    };
    let db = CoreNN::new_in_memory(cfg.clone());

    let (insert_throughput, insert_total_ms) = benchmark_insert(&db, &vectors);
    let (qps, mean, p50, p95, p99, _) = benchmark_queries(&db, &queries, k);

    info!(
      benchmark = name,
      insert_vps = insert_throughput,
      query_qps = qps,
      p50_us = p50,
      "completed"
    );

    results.push(BenchmarkResult {
      name: name.to_string(),
      dataset: "random".to_string(),
      dimension: dim,
      num_vectors,
      num_queries,
      k,
      metric: "L2".to_string(),
      insert_throughput_vps: insert_throughput,
      insert_total_ms,
      query_throughput_qps: Some(qps),
      query_latency_mean_us: Some(mean),
      query_latency_p50_us: Some(p50),
      query_latency_p95_us: Some(p95),
      query_latency_p99_us: Some(p99),
      recall_at_k: None,
      config: BenchmarkConfig {
        beam_width: cfg.beam_width,
        max_edges: cfg.max_edges,
        query_search_list_cap: cfg.query_search_list_cap,
        compression: None,
      },
    });
  }

  // Compression benchmarks on 768d/10k
  let compression_configs: Vec<(CompressionMode, &str, usize)> = vec![
    (CompressionMode::PQ, "pq", 16),
    (CompressionMode::SQ, "sq", 0),
  ];

  let dim = 768;
  let num_vectors = 10000;

  let vectors: Vec<Vec<f32>> = (0..num_vectors)
    .map(|_| random_f32_vec(&mut rng, dim))
    .collect();
  let queries: Vec<Vec<f32>> = (0..num_queries)
    .map(|_| random_f32_vec(&mut rng, dim))
    .collect();

  for (mode, mode_name, pq_subspaces) in compression_configs {
    let name = format!("random_768d_10k_{}", mode_name);
    info!(benchmark = %name, compression = mode_name, "starting");

    let cfg = Cfg {
      dim,
      metric: StdMetric::L2,
      beam_width: 4,
      max_edges: 32,
      query_search_list_cap: 128,
      update_search_list_cap: 128,
      compression_mode: mode,
      compression_threshold: 0,
      pq_subspaces: if pq_subspaces > 0 { pq_subspaces } else { 64 },
      ..Default::default()
    };
    let db = CoreNN::new_in_memory(cfg.clone());

    let (insert_throughput, insert_total_ms) = benchmark_insert(&db, &vectors);
    let (qps, mean, p50, p95, p99, _) = benchmark_queries(&db, &queries, k);

    info!(
      benchmark = %name,
      insert_vps = insert_throughput,
      query_qps = qps,
      p50_us = p50,
      "completed"
    );

    results.push(BenchmarkResult {
      name: name.clone(),
      dataset: "random".to_string(),
      dimension: dim,
      num_vectors,
      num_queries,
      k,
      metric: "L2".to_string(),
      insert_throughput_vps: insert_throughput,
      insert_total_ms,
      query_throughput_qps: Some(qps),
      query_latency_mean_us: Some(mean),
      query_latency_p50_us: Some(p50),
      query_latency_p95_us: Some(p95),
      query_latency_p99_us: Some(p99),
      recall_at_k: None,
      config: BenchmarkConfig {
        beam_width: cfg.beam_width,
        max_edges: cfg.max_edges,
        query_search_list_cap: cfg.query_search_list_cap,
        compression: Some(mode_name.to_string()),
      },
    });
  }

  results
}

fn run_dataset_benchmark(dataset_name: &str) -> Vec<BenchmarkResult> {
  let mut results = Vec::new();
  let datasets_dir = PathBuf::from(DATASETS_DIR).join("datasets");
  let dir = datasets_dir.join(dataset_name);

  let info_path = dir.join("info.toml");
  let info_str = std::fs::read_to_string(&info_path)
    .unwrap_or_else(|e| panic!("Failed to read {}: {}", info_path.display(), e));
  let info: DatasetInfo = toml::from_str(&info_str)
    .unwrap_or_else(|e| panic!("Failed to parse {}: {}", info_path.display(), e));

  info!(
    dataset = dataset_name,
    dtype = %info.dtype,
    metric = %info.metric,
    dim = info.dim,
    n = info.n,
    q = info.q,
    k = info.k,
    "loading dataset"
  );

  let vectors_path = dir.join("vectors.bin");
  let vectors = if info.dtype == "f16" {
    load_f16_vectors(&vectors_path, info.dim, info.n)
  } else {
    load_f32_vectors(&vectors_path, info.dim, info.n)
  };

  info!(dataset = dataset_name, vectors = vectors.len(), "loaded vectors");

  let metric = parse_metric(&info.metric);
  let metric_str = info.metric.clone();

  let has_groundtruth = info.q > 0 && info.k > 0;

  if has_groundtruth {
    let queries_path = dir.join("queries.bin");
    let results_path = dir.join("results.bin");

    let queries = if info.dtype == "f16" {
      load_f16_vectors(&queries_path, info.dim, info.q)
    } else {
      load_f32_vectors(&queries_path, info.dim, info.q)
    };
    let groundtruth = load_groundtruth(&results_path, info.q, info.k);

    info!(
      dataset = dataset_name,
      queries = queries.len(),
      groundtruth = groundtruth.len(),
      "loaded queries"
    );

    let k_values: Vec<usize> = vec![1, 10, info.k.min(100)];
    let search_caps = [128, 256];

    for &k in &k_values {
      for &search_cap in &search_caps {
        let bench_name = format!("{}_k{}_cap{}", dataset_name, k, search_cap);
        info!(benchmark = %bench_name, k = k, search_cap = search_cap, "starting");

        let cfg = Cfg {
          dim: info.dim,
          metric,
          beam_width: 4,
          max_edges: 32,
          query_search_list_cap: search_cap,
          update_search_list_cap: search_cap,
          compression_threshold: usize::MAX,
          ..Default::default()
        };
        let db = CoreNN::new_in_memory(cfg.clone());

        let (insert_throughput, insert_total_ms) = benchmark_insert(&db, &vectors);
        let (qps, mean, p50, p95, p99, query_results) = benchmark_queries(&db, &queries, k);
        let recall = compute_recall(&query_results, &groundtruth, k);

        info!(
          benchmark = %bench_name,
          insert_vps = insert_throughput,
          query_qps = qps,
          recall = recall,
          p50_us = p50,
          "completed"
        );

        results.push(BenchmarkResult {
          name: bench_name.clone(),
          dataset: dataset_name.to_string(),
          dimension: info.dim,
          num_vectors: vectors.len(),
          num_queries: queries.len(),
          k,
          metric: metric_str.clone(),
          insert_throughput_vps: insert_throughput,
          insert_total_ms,
          query_throughput_qps: Some(qps),
          query_latency_mean_us: Some(mean),
          query_latency_p50_us: Some(p50),
          query_latency_p95_us: Some(p95),
          query_latency_p99_us: Some(p99),
          recall_at_k: Some(recall),
          config: BenchmarkConfig {
            beam_width: cfg.beam_width,
            max_edges: cfg.max_edges,
            query_search_list_cap: cfg.query_search_list_cap,
            compression: None,
          },
        });
      }
    }
  } else {
    let bench_name = format!("{}_insert_only", dataset_name);
    info!(benchmark = %bench_name, "starting insert-only");

    let cfg = Cfg {
      dim: info.dim,
      metric,
      beam_width: 4,
      max_edges: 32,
      query_search_list_cap: 128,
      update_search_list_cap: 128,
      compression_threshold: usize::MAX,
      ..Default::default()
    };
    let db = CoreNN::new_in_memory(cfg.clone());

    let (insert_throughput, insert_total_ms) = benchmark_insert(&db, &vectors);

    info!(
      benchmark = %bench_name,
      insert_vps = insert_throughput,
      insert_ms = insert_total_ms,
      "completed"
    );

    results.push(BenchmarkResult {
      name: bench_name,
      dataset: dataset_name.to_string(),
      dimension: info.dim,
      num_vectors: vectors.len(),
      num_queries: 0,
      k: 0,
      metric: metric_str,
      insert_throughput_vps: insert_throughput,
      insert_total_ms,
      query_throughput_qps: None,
      query_latency_mean_us: None,
      query_latency_p50_us: None,
      query_latency_p95_us: None,
      query_latency_p99_us: None,
      recall_at_k: None,
      config: BenchmarkConfig {
        beam_width: cfg.beam_width,
        max_edges: cfg.max_edges,
        query_search_list_cap: cfg.query_search_list_cap,
        compression: None,
      },
    });
  }

  results
}

fn discover_datasets() -> Vec<String> {
  let datasets_dir = PathBuf::from(DATASETS_DIR).join("datasets");
  let mut datasets = Vec::new();

  if let Ok(entries) = std::fs::read_dir(&datasets_dir) {
    for entry in entries.flatten() {
      let path = entry.path();
      if path.is_dir() && path.join("info.toml").exists() && path.join("vectors.bin").exists() {
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
          datasets.push(name.to_string());
        }
      }
    }
  }

  datasets.sort();
  datasets
}

fn main() {
  tracing_subscriber::fmt().json().init();

  let args: Vec<String> = std::env::args().collect();

  let mut output_path: Option<PathBuf> = None;

  let mut i = 1;
  while i < args.len() {
    match args[i].as_str() {
      "--output" => {
        output_path = Some(PathBuf::from(&args[i + 1]));
        i += 2;
      }
      _ => {
        i += 1;
      }
    }
  }

  let mut all_results = Vec::new();

  info!("starting random benchmarks");
  all_results.extend(run_random_benchmarks());

  let datasets = discover_datasets();
  if datasets.is_empty() {
    info!("no datasets found");
  } else {
    info!(count = datasets.len(), "starting dataset benchmarks");
    for dataset in datasets {
      all_results.extend(run_dataset_benchmark(&dataset));
    }
  }

  let report = BenchmarkReport {
    commit: std::env::var("GITHUB_SHA").unwrap_or_else(|_| "unknown".to_string()),
    timestamp: chrono::Utc::now().to_rfc3339(),
    results: all_results,
  };

  let json = serde_json::to_string_pretty(&report).expect("Failed to serialize results");

  if let Some(path) = output_path {
    let mut file = File::create(&path).expect("Failed to create output file");
    file
      .write_all(json.as_bytes())
      .expect("Failed to write output file");
    info!(path = %path.display(), "results written");
  } else {
    println!("{}", json);
  }
}
