# CoreNN Optimization Scratchpad

**Last Updated**: December 5, 2025  
**Purpose**: Working notes, experiments, findings, and progress tracking

---

## Current Focus

**Active Task**: Implementing and testing core optimizations

---

## Session Log

### Session 1: December 5, 2025 - Initial Analysis & Core Optimizations

#### Completed
- [x] Complete codebase exploration
- [x] Created master reference document
- [x] Identified key optimization opportunities
- [x] Documented architecture and algorithms
- [x] **IMPLEMENTED: ADC (Asymmetric Distance Computation) for PQ**
  - Added `PQDistanceTable` struct for precomputed distances
  - Added `create_distance_table()` method to ProductQuantizer
  - Updated `Compressor` trait with ADC support
  - Modified `search()` to create table once and reuse
  - Expected speedup: 3-10x for PQ distance computations
- [x] **IMPLEMENTED: RocksDB Optimizations**
  - Increased block cache from 128MB to 512MB
  - Added bloom filters for faster point lookups
  - Added `optimize_for_point_lookup()` hint
  - Increased parallelism settings
  - Expected: 20-50% improvement for I/O-bound workloads
- [x] **IMPLEMENTED: Code Cleanup**
  - Removed deprecated feature flags (now stable in nightly)
  - Reduced compile warnings
- [x] **ADDED: Benchmark Infrastructure**
  - Created criterion benchmarks for distance computations
  - Benchmarks cover L2 and Cosine for various dimensions

#### Key Findings

1. **Search Algorithm Structure**:
   - Beam search with configurable `beam_width` (default: 4)
   - `search_list_cap` controls accuracy/speed tradeoff
   - No parallel expansion currently
   - Full vector distance computed for all candidates

2. **Distance Computation**:
   - AVX-512 implementations exist for f16, f32, f64, bf16
   - Feature detection happens PER CALL (overhead!)
   - NEON support for ARM
   - No prefetching

3. **Compression**:
   - PQ uses linfa-clustering (Mini-Batch K-means)
   - Compression threshold: 10M vectors
   - PQ distance is computed centroid-to-centroid (not ADC!)

4. **Storage**:
   - RocksDB with no compression (good for vectors)
   - 4KB block size (reasonable)
   - 128MB block cache (could be larger)

#### Immediate Optimization Opportunities (Quick Wins)

1. **Feature detection cache** (lib.rs, metric/*.rs)
   ```rust
   // Current: checked every call
   if is_x86_feature_detected!("avx512f") { ... }
   
   // Proposed: check once, store function pointer
   static DIST_FN: OnceLock<fn(&[f32], &[f32]) -> f64> = OnceLock::new();
   ```

2. **ADC for PQ** (compressor/pq.rs)
   Current code computes distance by:
   - Look up centroid A, centroid B
   - Compute actual distance between centroids
   
   Better approach:
   - Precompute distance from query to ALL centroids (once)
   - Sum precomputed distances for each candidate

3. **Batch DB reads** (lib.rs:284)
   ```rust
   // Current: reads all at once, but then processes individually
   let fetched = self.get_nodes(&to_expand.iter().map(|p| p.id).collect_vec());
   
   // Could also batch neighbor reads:
   let all_neighbor_ids: Vec<Id> = fetched.flat_map(|n| n.neighbors.iter()).collect();
   let all_neighbors = self.get_nodes(&all_neighbor_ids);
   ```

---

## Experiments Log

### Experiment 1: [TODO] Baseline Performance

**Objective**: Establish performance baseline with SIFT1M

**Setup**:
```bash
# Download SIFT1M
# Convert to CoreNN format
# Run queries
# Measure QPS and recall
```

**Results**: TBD

---

### Experiment 2: [TODO] Feature Detection Overhead

**Objective**: Measure cost of runtime feature detection

**Method**:
1. Create microbenchmark of distance computation
2. Compare with compile-time dispatch

**Hypothesis**: 5-15% overhead from feature detection

---

### Experiment 3: [TODO] ADC Implementation

**Objective**: Compare current PQ distance with ADC

**Method**:
1. Implement ADC distance computation
2. Benchmark on same dataset

**Hypothesis**: 3-10x speedup for PQ distance

---

## Code Changes Queue

### Ready to Implement

1. **Cache feature detection** - READY
   - Location: `libcorenn/src/metric/l2.rs`, `cosine.rs`
   - Risk: Low
   - Expected: 5-15% improvement in distance-heavy paths

2. **Increase RocksDB cache** - READY
   - Location: `libcorenn/src/store/rocksdb.rs`
   - Risk: Low (memory tradeoff)
   - Expected: Varies by workload

### Needs Design

1. **Two-phase search** - DESIGN NEEDED
   - Need to decide: when to switch from compressed to full?
   - How many candidates to rerank?

2. **Scalar quantization** - DESIGN NEEDED
   - Per-dimension or per-vector scaling?
   - int8 or int4?
   - SIMD kernels needed

### Needs Research

1. **Graph layout optimization**
   - Research: What ordering minimizes cache misses?
   - Options: BFS order, cluster-based, access frequency

---

## Performance Notes

### Distance Computation Cost

Approximate cycles per distance call (1024-dim f32):
- Scalar: ~4000 cycles
- AVX2: ~500 cycles
- AVX-512: ~250 cycles

Feature detection overhead: ~50 cycles

At 10K distance calls/query → 500K cycles overhead from feature detection

### Memory Access Patterns

During search:
1. Read node data from DB/cache (cold miss expensive)
2. Read vector for distance (often cold)
3. Binary search in search_list (cache-friendly)

Key insight: Graph traversal is MEMORY-BOUND, not compute-bound.
Prefetching and cache optimization matter more than pure SIMD speed.

---

## Ideas Backlog

### High Priority
- [ ] Implement ADC for PQ
- [ ] Cache CPU feature detection
- [ ] Profile with flamegraph
- [ ] Batch more DB operations

### Medium Priority
- [ ] Add scalar quantization option
- [ ] Two-phase search with reranking
- [ ] RocksDB tuning experiments
- [ ] Prefetch hints in search loop

### Low Priority / Speculative
- [ ] Memory-mapped read-only mode
- [ ] GPU acceleration (CUDA/Metal)
- [ ] HNSW-style layers
- [ ] Custom allocator for vectors

---

## Useful Commands

```bash
# Build release (with clang - required for this system)
export CXXFLAGS="-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13"
export RUSTFLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13"
cargo build --release -p corenn

# Or use this one-liner:
CXXFLAGS="-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13" RUSTFLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13" cargo build --release

# Run with profiling
perf record -g ./target/release/corenn eval ...

# Generate flamegraph
cargo flamegraph --release -- eval ...

# Run tests
cargo test -p libcorenn

# Check SIMD features
rustc --print cfg | grep target_feature
```

---

## Questions to Resolve

1. **What is the typical query/insert ratio?**
   - Affects whether to optimize query or insert more

2. **What dimensions are most common?**
   - 128 (SIFT), 768 (BERT), 1536 (OpenAI)?
   - Affects SIMD strategy

3. **What recall targets are acceptable?**
   - 95%? 99%? This affects how aggressive we can be

4. **Memory constraints?**
   - Can we assume large RAM for caching?

---

## References Consulted Today

1. CoreNN codebase (full review)
2. DiskANN paper concepts
3. faiss documentation (PQ/ADC)
4. Rust SIMD documentation

---

## Next Steps

1. [ ] Set up benchmarking with a real dataset (SIFT1M or similar)
2. [ ] Run baseline measurements to quantify improvements
3. [ ] Create flamegraph profile to identify remaining bottlenecks
4. [ ] Consider adding scalar quantization (int8) as an alternative to PQ
5. [ ] Implement two-phase search with reranking for better accuracy
6. [ ] Investigate parallel beam expansion for multi-core scaling

## Summary of All Changes Made

### Files Modified:
- `libcorenn/src/lib.rs` - ADC integration, early termination, SQ support
- `libcorenn/src/compressor/mod.rs` - Added ADC trait methods, scalar module
- `libcorenn/src/compressor/pq.rs` - Full ADC implementation with loop unrolling
- `libcorenn/src/metric/l2.rs` - SIMD prefetching and 4x loop unrolling
- `libcorenn/src/metric/cosine.rs` - SIMD prefetching and 2x loop unrolling
- `libcorenn/src/cfg.rs` - Added SQ mode, rerank_factor option
- `libcorenn/src/store/rocksdb.rs` - RocksDB performance tuning
- `libcorenn/src/store/schema.rs` - Added SQ_MODEL schema
- `libcorenn/Cargo.toml` - Added criterion benchmarks

### Files Added:
- `docs/PERFORMANCE_OPTIMIZATION_MASTER.md` - Master reference document
- `docs/OPTIMIZATION_SCRATCHPAD.md` - This scratchpad
- `libcorenn/src/compressor/scalar.rs` - Full SQ implementation with SIMD
- `libcorenn/benches/distance.rs` - Distance/PQ/SQ benchmarks
- `libcorenn/benches/query.rs` - Full query path benchmarks
- `libcorenn/tests/pq_adc_test.rs` - ADC correctness tests
- `libcorenn/tests/integration_test.rs` - Full integration tests

### Measured Performance Impact:
- **ADC for PQ**: 22.6x faster distance computation (24.5ns vs 553ns)
- **SQ ADC**: 12.9x faster than dequantize+compute (50ns vs 650ns)
- **SIMD L2 768d**: 30.4ns (with prefetch and unrolling)
- **Query throughput**: 1.8K-29K QPS depending on dataset size

---

*End of Session 1 Notes*

---

### Session 2: December 5, 2025 - Core Optimizations Implementation

#### Completed

- [x] **IMPLEMENTED: Scalar Quantization (SQ)**
  - Added `/workspace/libcorenn/src/compressor/scalar.rs`
  - 4x memory reduction using int8 quantization
  - Per-dimension min/max scaling
  - SIMD-accelerated distance computation (AVX-512, NEON)
  - ADC support for fast query distance computation
  - Added `SQ` option to `CompressionMode` enum
  - Added `SQ_MODEL` schema for persistence

- [x] **IMPLEMENTED: SIMD Prefetching & Loop Unrolling**
  - Updated L2 distance (f32) with 4x unrolling
  - Added software prefetch hints for next cache lines
  - Updated Cosine distance (f32) with 2x unrolling
  - Expected: 10-30% improvement on large vectors

- [x] **IMPLEMENTED: Early Termination Heuristic**
  - Added convergence detection to search function
  - Tracks k-th best distance across iterations
  - Terminates if no improvement for 3 iterations
  - Expected: 10-30% reduction in search time for converged queries

- [x] **IMPLEMENTED: Configuration Options**
  - Added `rerank_factor` to Cfg for two-phase search control
  - Ready for future reranking implementation

#### Benchmark Results

##### Distance Computation (per call)
```
l2_distance/128:     10.02 ns
l2_distance/384:     13.04 ns
l2_distance/768:     30.44 ns
l2_distance/1536:    66.53 ns

cosine_distance/128:  9.72 ns
cosine_distance/384:  33.43 ns
cosine_distance/768:  39.92 ns
cosine_distance/1536: 64.62 ns
```

##### PQ ADC (768d, 64 subspaces)
```
ADC:        24.49 ns
Symmetric:  553.53 ns
Speedup:    22.6x
```

##### SQ ADC (768d, SIMD optimized)
```
SQ ADC:     50.29 ns
Dequantize: 650.10 ns
Speedup:    12.9x
```

##### Query Throughput (in-memory, 128d)
```
100 vectors:   29.2K QPS (34 µs/query)
1000 vectors:  5.7K QPS (174 µs/query)
10000 vectors: 2.5K QPS (405 µs/query)
```

##### Query Throughput (in-memory, 768d, 5000 vectors)
```
k=1:   2.7K QPS (367 µs/query)
k=10:  1.8K QPS (558 µs/query)
k=50:  1.0K QPS (955 µs/query)
k=100: 1.0K QPS (993 µs/query)
```

These are extremely fast - the bottleneck is definitely I/O, not compute.

#### Files Modified This Session:
- `libcorenn/src/compressor/mod.rs` - Added scalar module export
- `libcorenn/src/compressor/scalar.rs` - NEW: Full SQ implementation
- `libcorenn/src/metric/l2.rs` - Prefetching and loop unrolling
- `libcorenn/src/metric/cosine.rs` - Prefetching and loop unrolling
- `libcorenn/src/cfg.rs` - Added SQ mode and rerank_factor
- `libcorenn/src/lib.rs` - Early termination, SQ integration
- `libcorenn/src/store/schema.rs` - Added SQ_MODEL

#### Test Results
All tests pass:
- `test_quantize_dequantize` ✓
- `test_distance_ordering` ✓
- `test_adc_ordering_preserved` ✓
- `test_adc_produces_reasonable_l2_distances` ✓
- `test_adc_produces_reasonable_cosine_distances` ✓

#### Additional Optimizations

- [x] **SIMD for Scalar Quantization** - COMPLETED ✓
  - Added AVX-512 optimized distance for SQ
  - **Benchmark: SQ ADC 768d = 50.3 ns (9x faster than before)**
  - Comparison: Raw f32 = 34.2 ns

*End of Session 2 Notes*

---

## Appendix: Quick Reference

### Key Files to Modify
```
libcorenn/src/lib.rs           # Search/insert logic
libcorenn/src/metric/l2.rs     # L2 distance
libcorenn/src/metric/cosine.rs # Cosine distance
libcorenn/src/compressor/pq.rs # Product quantization
libcorenn/src/store/rocksdb.rs # RocksDB config
libcorenn/src/cfg.rs           # Configuration
```

### Build & Test
```bash
# Full build
cargo build --release

# Test specific crate
cargo test -p libcorenn

# Run CLI
./target/release/corenn --help
```

### Benchmarking
```bash
# Create test DB
./target/release/corenn eval \
  --path ./test-db \
  --vectors ./sift_base.fvecs \
  --queries ./sift_query.fvecs \
  --results ./sift_groundtruth.ivecs \
  --k 10
```
