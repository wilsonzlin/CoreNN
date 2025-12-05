# CoreNN Performance Improvement Playbook

> Living reference for multi-day optimization work. Keep this document authoritative; record every major decision, data point, and insight here.

## 1. Mission & Success Criteria
- **Primary goal**: reduce average and tail query latency while sustaining/improving recall for billion-scale vector datasets on commodity hardware.
- **Secondary goals**: increase ingestion throughput, reduce storage cost (RAM + disk), and preserve API ergonomics across Rust/Python/Node bindings.
- **Success metrics** (measure per dataset + hardware target):
  - Query latency (P50/P95/P99) at `k ∈ {10, 100}` for representative workloads.
  - Throughput (queries/sec) with multi-threaded clients.
  - Recall@k against exact ground truth (`corenn cmd eval`).
  - Build/ingest throughput (vectors/sec) for batch + streaming insert paths.
  - Index size (GB) and steady-state RSS during query.
- **Constraints**: preserve correctness, remain self-hostable, avoid regressions for both RocksDB-backed and in-memory stores, keep ergonomic config surface (`libcorenn/src/cfg.rs`).

## 2. Current System Snapshot (Dec 2025)
### Components
- `libcorenn/`: core Rust library implementing graph-based ANN search and storage abstractions.
- `corenn/`: CLI for eval, export, migration, and HTTP serving (see `corenn/src/cmd/*.rs`).
- `corenn-py/` & `corenn-node/`: bindings for Python and Node ecosystems.
- `hnswlib-rs/`: reference/compat layer for migrations.
- `tools/`: Python scripts for dataset prep + analysis (e.g., `tools/calc_nn.py`).

### Key Architectural Notes
- Graph index resembles a single-layer navigable small-world graph with configurable beam width and candidate cap (`libcorenn/src/lib.rs::search`). No multi-layer hierarchy yet (contrast with HNSW).
- Storage via trait `Store` with RocksDB + in-memory implementations. RocksDB tuned with 4KB block size and async I/O to mimic FreshDiskANN block layout.
- Compression pipeline supports Product Quantization (`CompressionMode::PQ`) and Matryoshka-style truncation (`CompressionMode::Trunc`), gated by `cfg.compression_threshold`.
- Concurrency: heavy use of `DashMap`, `DashSet`, `parking_lot` locks, and `rayon` for parallel inserts and compaction.
- Compaction rebalances neighbor sets after deletes, inspired by FreshDiskANN (see `libcorenn/src/compaction.rs`).
- CLI uses `tokio` runtime + `rayon` for data parallel loops; `tikv_jemallocator` is the global allocator to avoid glibc fragmentation.

## 3. Baseline & Instrumentation Plan
1. **Datasets** (pick at least one per scale):
   - `SIFT1M` (128 dims) for rapid iteration.
   - `Deep1B` / `BigANN-1B` (96 dims) for billion-scale realism.
   - Domain-specific (e.g., sentence embeddings) if provided by stakeholders.
2. **Ground truth**: use FAISS exact search or `tools/calc_nn.py` to precompute u32 neighbor lists consumed by `corenn cmd eval`.
3. **Benchmark harness**:
   - Use `corenn cmd eval --vectors --queries --results` to ingest + benchmark from packed matrices.
   - Wrap with Criterion or a custom Rust harness to sample latency distributions; log to JSON for trend tracking.
   - Automate multi-threaded client scenarios via the HTTP server (`corenn::cmd::serve`) and a load generator (e.g., `hey`, `wrk`, or custom Rust client).
4. **Profiling**:
   - `perf record`, `cargo flamegraph`, and `tokio-console` for async hotspots.
   - `jemalloc` heap profiling (`MALLOC_CONF=prof:true`) to catch leaks or fragmentation.
   - RocksDB metrics via `get_statistics_string` if enabled.
5. **Hardware matrix**: minimum of (a) developer workstation (32-core/64GB), (b) production-like server (>= 64 cores, NVMe). Document CPU model, RAM, storage, kernel, and compiler flags.
6. **Change management**: capture every run in a spreadsheet or JSON log containing git SHA, config, dataset, metrics, hardware, and notes. Store pointer in this document.

## 4. Opportunity Areas & Hypotheses
| Pillar | Observations | Candidate improvements |
| --- | --- | --- |
| Graph search | Single-layer beam search may require large `search_list_cap` to maintain recall. No dynamic entry point selection beyond node 0. | Explore multi-layer hierarchy (HNSW-like), Vamana/NSG neighbor diversification, entry-point heuristics (randomized or medoid), adaptive beam width, visited-set bitsets.
| Candidate pruning | Current pruning (`distance_threshold`, `max_edges`) mirrors heuristic from DiskANN but re-computes distances frequently. | Precompute neighbor distances when possible, use SIMD distance kernels (AVX-512) with SoA layout, adopt heuristic updates like greedy graph augmentation.
| Compression | PQ parameters fixed at runtime; compression triggered at threshold with DB roundtrips. | Auto-tune `pq_subspaces`, consider Optimized Product Quantization (OPQ) and scalar quantization (int8/int4), integrate residual quantization for 2-stage search.
| Storage / RocksDB | Default options moderate; no tiered compaction strategy or column families. | Evaluate separate column families per entity, larger block cache, BlobDB for vectors, range-deletion for expired nodes, prefetch neighbors sequentially.
| Memory layout | `VecData` stores heap-allocated Vecs per node causing pointer chasing. | Investigate contiguous arena for uncompressed vectors, align data for SIMD, convert to `Arc<[f32]>` slices to reduce refcount overhead.
| Parallelism | Query path mostly single-threaded aside from DashMap concurrency. | Batch queries for SIMD, multi-query search sharing candidate expansions, GPU offload for distance calculations.
| API ergonomics | Config lacks auto-tuning guidance. | Provide `Cfg::profile_*` presets, command-line auto-tuner, or runtime adaptation based on dataset stats.

## 5. Research & External References
- **Graph ANN**: HNSW, NSG, Vamana, Hierarchical Navigable Small Worlds, DiskANN, FreshDiskANN, PANNG, LSH graphs, GraphANN.
- **Compression & Quantization**: OPQ, Product Quantization, IVF-PQ, Scalar Quantization, Matryoshka embeddings (for truncation mode), PQFastScan, Bolt (vector compression for search), NVIDIA Raft/RAFT cuVS.
- **Systems**: FAISS, ScaNN, Qdrant, Milvus, Weaviate, Quickwit, Pinecone, DiskANN (Microsoft), FreshDiskANN.
- **Hardware-specific**: SIMD optimized distance kernels (FAISS fvec), GPU-based ANN (RAFT cuVS, FAISS-GPU), DPUs or BlueField for offloading.
- **RocksDB tuning**: Level style vs FIFO compaction, partitioned indexes/filters, `ReadOptions::set_readahead_size`, `BlockBasedTableOptions::set_data_block_hash_index_type`.
Document key takeaways from each paper/tool inside this file as research completes.

## 6. Execution Roadmap (Iterate but keep disciplined)
1. **Orientation & Baseline (Day 0-1)**
   - Build database from reference dataset, collect baseline metrics for default config.
   - Capture flamegraphs for query + insert.
   - Produce dashboard/spreadsheet for metrics; store sample DB artifacts for regression tests.
2. **Low-hanging Optimizations (Day 2-4)**
   - Parameter sweeps (`beam_width`, `query_search_list_cap`, PQ params) using automated harness.
   - Optimize hot loops: inline `Point::dist_query`, switch to SoA, prefetch neighbor vectors, reduce allocations in `search`.
   - Add instrumentation counters (cache hits, DB roundtrips, candidate counts) exposed via tracing.
3. **Engine Revamp (Day 5-10)**
   - Prototype multi-entry search (randomized restarts) and evaluate recall/speed.
   - Evaluate hierarchical graph build akin to HNSW or Vamana's directed acyclic graph.
   - Add optional disk-resident distance cache or neighbor prefetch queue.
4. **Compression & Storage (Day 8-14)**
   - Implement OPQ training path; add CLI to train + persist.
   - Experiment with scalar quantization for `VecData::F16` ingestion.
   - Investigate RocksDB column families or Sled/Bonsai alternatives; benchmark block cache sizing.
5. **Advanced Features (Day 12+)**
   - Batch query API + AVX512 distance kernels.
   - GPU distance backend via CUDA or Vulkan compute (optional feature gate).
   - Adaptive auto-tuner that picks config based on dataset stats.
6. **Stabilization**
   - Regression suite, soak tests, failure injection (kill compaction mid-run), durability tests.
   - Document tuning guides, update SDK docs.

## 7. Experiment Log Template
Maintain a table (append new rows) for every experiment:

| Date | Git SHA | Dataset | Config | Change | Recall@k | Lat P50/P99 | QPS | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Store the detailed JSON/CSV artifacts under `docs/benchmarks/YYYY-MM-DD/` (create as needed).

## 8. Risks & Open Questions
- **Data consistency**: concurrent insert/delete vs compaction interactions—need fuzz tests.
- **RocksDB latency variance**: asynchronous reads + background compactions could dominate tail latency.
- **Memory pressure**: caches for uncompressed and compressed vectors may duplicate data; need adaptive eviction policy.
- **Compression accuracy**: PQ error may harm recall for high dimensions; may require hybrid approach (exact re-ranking using full vectors stored separately).
- **API compatibility**: introducing multi-layer graphs or new compressors must remain backward compatible with stored DBs or offer migration tooling.
- **GPU integration**: ensures zero-copy data movement and compatibility with existing CPU fallback.

## 9. Collaboration & Documentation Norms
- Keep this file updated whenever new discoveries are made.
- Use the scratchpad (`docs/performance_scratchpad.md`) for transient calculations, TODOs, and hypotheses; periodically distill important findings back here.
- When implementing, reference relevant sections here in commit messages for traceability.
- Prefer deterministic, scripted benchmarks so results are reproducible by the team and future contributors.
