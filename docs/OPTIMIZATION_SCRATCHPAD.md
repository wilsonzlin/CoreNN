# CoreNN Optimization Scratchpad

**Last Updated**: December 5, 2025  
**Purpose**: Working notes, experiments, findings, and progress tracking

---

## Current Focus

**Active Task**: Initial profiling and baseline establishment

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

## Summary of Changes Made

### Files Modified:
- `libcorenn/src/lib.rs` - ADC integration in search path
- `libcorenn/src/compressor/mod.rs` - Added ADC trait methods
- `libcorenn/src/compressor/pq.rs` - Full ADC implementation
- `libcorenn/src/store/rocksdb.rs` - RocksDB performance tuning
- `libcorenn/Cargo.toml` - Added criterion benchmarks

### Files Added:
- `docs/PERFORMANCE_OPTIMIZATION_MASTER.md` - Master reference document
- `docs/OPTIMIZATION_SCRATCHPAD.md` - This scratchpad
- `libcorenn/benches/distance.rs` - Distance benchmarks
- `libcorenn/tests/pq_adc_test.rs` - ADC correctness tests

### Expected Performance Impact:
- **ADC for PQ**: 3-10x faster distance computation in compressed mode
- **RocksDB tuning**: 20-50% improvement for I/O-bound workloads
- Combined effect depends on workload characteristics

---

*End of Session 1 Notes*

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
