# CoreNN Internal Engineering Reference

Billion-scale vector database for ANN search. DiskANN/Vamana graph algorithm with RocksDB persistence, PQ/SQ compression, SIMD distance computation.

## Architecture

```
libcorenn/src/
├── lib.rs          Core CoreNN struct, search/insert logic
├── cfg.rs          Configuration (hyperparameters)
├── cache.rs        In-memory node caching
├── compaction.rs   Graph maintenance, delete handling
├── common.rs       Common types (Id)
├── util.rs         Atomic utilities
├── vec.rs          VecData (bf16/f16/f32/f64)
├── metric/
│   ├── mod.rs      Metric trait
│   ├── l2.rs       L2 distance (SIMD)
│   └── cosine.rs   Cosine distance (SIMD)
├── compressor/
│   ├── mod.rs      Compressor trait
│   ├── pq.rs       Product Quantization
│   ├── scalar.rs   Scalar Quantization
│   └── trunc.rs    Truncation (Matryoshka)
└── store/
    ├── mod.rs      Store trait
    ├── rocksdb.rs  RocksDB backend
    ├── in_memory.rs In-memory backend
    └── schema.rs   DB schema (NODE, ADD_EDGES, etc.)
```

## Core Data Structures

### DbNodeData (store/schema.rs)
```rust
pub struct DbNodeData {
  pub version: u64,           // Incremented on update, used for cache invalidation
  pub neighbors: Vec<Id>,     // Graph edges
  pub vector: Arc<VecData>,   // Co-located with neighbors (one disk page read)
}
```

### VecData (vec.rs)
```rust
pub enum VecData {
  BF16(Vec<bf16>),
  F16(Vec<f16>),
  F32(Vec<f32>),
  F64(Vec<f64>),
}
```

### State (lib.rs)
```rust
pub struct State {
  add_edges: DashMap<Id, Vec<Id>>,    // Pending backedges (lazy pruning)
  cfg: Cfg,                            // Immutable after creation
  db: Arc<dyn Store>,                  // RocksDB or InMemory
  deleted: DashSet<Id>,                // Soft-deleted IDs
  mode: RwLock<Mode>,                  // Uncompressed or Compressed
  count: AtomUsz,                      // Vector count
  next_id: AtomUsz,                    // ID allocator
  // ...caches, locks
}
```

### Mode (lib.rs)
```rust
enum Mode {
  Uncompressed(NodeCache),             // Lazy cache of DbNodeData in-memory
  Compressed(Arc<dyn Compressor>, CVCache),  // PQ/SQ/Trunc compressed vectors
}
```

### Database Schema (store/schema.rs)
```
ADD_EDGES: Id → Vec<Id>           Pending edges for lazy updates
CFG: () → Cfg                     Configuration
DELETED: Id → ()                  Soft-deleted IDs
KEY_TO_ID: String → Id            String key to numeric ID
ID_TO_KEY: Id → String            Numeric ID to string key
NODE: Id → DbNodeData             Graph nodes with vectors and edges
PQ_MODEL: () → ProductQuantizer   Product Quantization model
SQ_MODEL: () → ScalarQuantizer    Scalar Quantization model
```

## Configuration (cfg.rs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | required | Vector dimensionality |
| `metric` | L2 | L2 or Cosine |
| `beam_width` | 4 | Nodes expanded per search iteration |
| `max_edges` | 64 | Max neighbors per node |
| `max_add_edges` | 128 | Pending edges before lazy pruning triggers |
| `distance_threshold` | 1.2 | α for Vamana RobustPrune (controls graph density) |
| `query_search_list_cap` | 128 | Search list size for queries |
| `update_search_list_cap` | 128 | Search list size for inserts |
| `compression_mode` | PQ | PQ, SQ, or Trunc |
| `compression_threshold` | 10M | Enable compression after N vectors |
| `pq_subspaces` | 64 | PQ subspace count |
| `pq_sample_size` | 10K | PQ training sample size |
| `trunc_dims` | 64 | Truncation dimensions (Matryoshka) |

## Algorithms

### Search (lib.rs)
Greedy beam search with HNSW-style early stopping:

```
1. Start from entry node (id=0, clone of first inserted vector)
2. Initialize lower_bound = entry.distance
3. Maintain search_list sorted by distance (max size: search_list_cap)
4. Loop:
   a. Pop beam_width unexpanded nodes from search_list
   b. Early stop: if best_unexpanded > lower_bound AND list is full: break
   c. For each expanded node:
      - Fetch neighbors from DB (NODE) + pending edges (add_edges)
      - Add unseen neighbors to search_list (only if dist < lower_bound)
      - Re-rank expanded node with full vector distance
   d. Truncate search_list to search_list_cap
   e. Update lower_bound = worst result distance
5. Return top-k from search_list
```

### Insert (lib.rs)
```
1. Assign id = next_id++
2. candidates = search(vector, k=1, update_search_list_cap)
3. neighbors = prune_candidates(vector, candidates)  // Vamana RobustPrune
4. Save node (id, neighbors, vector) to DB
5. For each neighbor j:
   - If j.add_edges.len >= max_add_edges:
       j.neighbors = prune_candidates(j.vector, j.neighbors + j.add_edges)
       Save j to DB
   - Else:
       j.add_edges.append(id)
```

### Vamana RobustPrune (lib.rs)
Algorithm 2 from DiskANN paper (Subramanya et al., NeurIPS 2019):

```
RobustPrune(p, V, α, R):
  V ← (V ∪ Nout(p)) \ {p}    // Merge with existing neighbors
  Nout(p) ← ∅

  while V ≠ ∅ do
    p* ← argmin_{p' ∈ V} d(p, p')    // Pick closest to node p
    Nout(p) ← Nout(p) ∪ {p*}          // Add to neighbors
    if |Nout(p)| = R then break       // Stop at max degree

    for p' ∈ V do
      if α · d(p*, p') ≤ d(p, p') then  // α-RNG condition
        remove p' from V                 // Prune covered points
```

The α parameter (distance_threshold) is CRUCIAL:
- α = 1.0: Standard RNG, sparser graph, potentially larger diameter
- α > 1.0: Denser graph, **guarantees O(log n) diameter** for disk-based search
- α = 1.2: Recommended value (DiskANN paper)

Each search step makes multiplicative progress: `d(query, next) ≤ d(query, current) / α`

Complexity: O(R × |V|) where R = max_edges, |V| = candidates

### Compaction (compaction.rs)
Handles deleted vectors. Iterates all nodes to remove edges to deleted nodes. Uses RocksDB snapshot for consistent iteration during concurrent updates.

## Compression

### Product Quantization (compressor/pq.rs)
Subspace decomposition: D dimensions → M subspaces × 256 centroids

Training: Mini-Batch K-means via linfa-clustering on sampled vectors

Encoding: Each subspace maps to 1-byte centroid index → M bytes per vector

ADC (Asymmetric Distance Computation):
1. Query stays uncompressed
2. Precompute distance from query subvector to all 256 centroids per subspace
3. For each compressed vector: sum table lookups (O(M) vs O(D))

PQDistanceTable:
```rust
struct PQDistanceTable {
  squared_distances: Vec<[f32; 256]>,  // L2: query to centroids per subspace
  dot_products: Vec<[f32; 256]>,       // Cosine: for dot product computation
  query_norms_sq: Vec<f32>,            // Cosine: query norm per subspace
  centroid_norms_sq: Vec<[f32; 256]>,  // Cosine: centroid norms
  metric: StdMetric,
}
```

### Scalar Quantization (compressor/scalar.rs)
Per-dimension quantization to int8:
```
q = round((x - min) / (max - min) * 255)
```

Training: Compute per-dimension min/max from sample vectors

4x memory reduction. SIMD-friendly (AVX-512, NEON).

SQDistanceTable:
```rust
struct SQDistanceTable {
  scaled_query: Vec<f32>,   // (query - min) * scale per dimension
  metric: StdMetric,
  query_norm_sq: f32,       // For cosine
}
```

### Truncation (compressor/trunc.rs)
For Matryoshka embeddings. Simply truncates to first N dimensions.

## Distance Computation (metric/)

Supported metrics: L2, Cosine

SIMD implementations:
- AVX-512 (x86): 16 f32 simultaneously, VDPBF16PS for bf16
- AVX-512 FP16: 32 f16 native
- NEON (ARM): 4 f32

Optimizations applied:
- 4x loop unrolling for L2
- 2x loop unrolling for Cosine
- Software prefetch hints

## Performance Benchmarks

### Distance Computation (per call)
| Dimension | L2 (f32) | Cosine (f32) |
|-----------|----------|--------------|
| 128 | 10.0 ns | 9.7 ns |
| 384 | 13.0 ns | 33.4 ns |
| 768 | 30.4 ns | 39.9 ns |
| 1536 | 66.5 ns | 64.6 ns |

### PQ ADC (768d, 64 subspaces)
| Method | Time | Speedup |
|--------|------|---------|
| ADC | 24.5 ns | 22.6x |
| Symmetric | 553.5 ns | baseline |

### SQ ADC (768d)
| Method | Time | Speedup |
|--------|------|---------|
| SQ ADC | 50.6 ns | 13.4x |
| Dequantize+Compute | 676.7 ns | baseline |

### Query Throughput (in-memory, uncompressed)
| Dataset | k | QPS |
|---------|---|-----|
| 128d, 100 vecs | 10 | 31.5K |
| 128d, 1K vecs | 10 | 8.4K |
| 128d, 10K vecs | 10 | 650 |
| 768d, 5K vecs | 10 | 537 |

## RocksDB Configuration (store/rocksdb.rs)
- Block cache: 512MB
- Bloom filters enabled
- Point lookup optimization hint
- Increased parallelism
- No compression (vectors don't compress well)

## CI Benchmarking

### Workflow (.github/workflows/benchmark.yml)
Matrix-based parallel jobs for each dataset.

### Datasets (hosted at https://static.wilsonl.in/embedding-datasets/)
| Dataset | Vectors | Dims | Metric | Ground Truth |
|---------|---------|------|--------|--------------|
| siftsmall | small | 128 | L2 | yes |
| sift-250k | 250K | 128 | L2 | yes |
| sift | 1M | 128 | L2 | yes |
| gist-250k | 250K | 960 | L2 | yes |
| gist | 1M | 960 | L2 | yes |
| bbcnews-nomicembed15 | varies | 768 | Cosine | no |
| bbcnews-static256 | varies | 256 | Cosine | no |
| steam-games | varies | varies | varies | no |
| gdelt-us-news | varies | varies | varies | no |

Dataset format:
- info.toml: `{dtype, metric, dim, n, q, k}`
- vectors.bin: packed little-endian matrix (n × dim × sizeof(dtype))
- queries.bin: packed queries (q × dim × sizeof(dtype))
- results.bin: ground truth u32 indices (q × k × 4)

Datasets without q/k run insert-only benchmarks.

### CI Binary (ci/)
```
ci --output results.json
```

Runs:
1. Random vector benchmarks (128d-1536d, 1K-50K vectors)
2. Compression benchmarks (PQ, SQ on 768d/10K)
3. Dataset benchmarks (auto-discovered from datasets/ folder)

Output: JSON with insert throughput, query QPS, latency percentiles, recall@k

## Vamana vs HNSW

| Aspect | Vamana/DiskANN | HNSW |
|--------|----------------|------|
| Structure | Single layer | Multi-layer skip-list |
| Entry point | Fixed node 0 | Top layer node |
| Pruning condition | α · d(p*, p') ≤ d(p, p') | d(p*, p') < d(q, p') |
| α parameter | Yes (controls diameter) | No |
| Theoretical guarantee | O(log n) with α > 1 | No formal bound |
| Insert speed | ~2K/sec | ~10K/sec |
| Memory | ~0.8KB/vector (128d, R=64) | ~1.2KB/vector (128d, M=16) |
| Best for | Disk-based, read-heavy | In-memory, write-heavy |

CoreNN uses Vamana because:
1. α parameter guarantees bounded latency for disk systems
2. Lower memory footprint
3. Single-layer simplifies persistence

## Tuning Guide

### For higher recall
- Increase `query_search_list_cap` (200-400 for 95%+, 400-600 for 99%+)
- Increase `beam_width` (8-16)
- Increase `max_edges` (96-128)
- Increase `distance_threshold` (α = 1.3-1.5)

### For higher speed
- Decrease `query_search_list_cap` (64-100)
- Decrease `beam_width` (2-4)
- Decrease `max_edges` (32-48)
- Decrease `distance_threshold` (α = 1.1)

### For memory reduction
- Use SQ (4x reduction) or PQ (16-32x reduction)
- Lower `max_edges`
- Lower `compression_threshold`

## API

### Rust
```rust
let db = CoreNN::create("/path/to/db", Cfg { dim: 768, ..Default::default() });
db.insert("key", &vec);
let results = db.query(&query, 100);  // Vec<(String, f64)>

// Open existing
let db = CoreNN::open("/path/to/db");
```

### Python
```python
from corenn_py import CoreNN
db = CoreNN.create("/path/to/db", {"dim": 768})
db.insert_f32(keys, vectors)  # vectors: numpy array
results = db.query_f32(queries, 100)  # list of list of (key, dist)
```

### Node.js
```typescript
import { CoreNN } from "@corenn/node";
const db = CoreNN.create("/path/to/db", { dim: 768 });
db.insert([{ key: "k1", vector: new Float32Array([...]) }]);
const results = db.query(queryVec, 100);  // { key, distance }[]
```

## Key Implementation Details

### Lazy Pruning
`add_edges` accumulates backedges. Pruning triggers when `add_edges.len >= max_add_edges` (default 128 = 2x max_edges). Reduces write amplification.

### Entry Point
Always node 0 (clone of first inserted vector). Static, unlike HNSW's dynamic top-layer entry.

### Cache
Lazy in-memory cache of DbNodeData (uncompressed mode) or compressed vectors (compressed mode). Avoids DB roundtrips, not computation.

### Visited Tracking
DashSet per search. TODO: visited list pool with generation counter for high QPS.

### Concurrency
- `DashMap` for add_edges (lock-free reads)
- `ArbitraryLock` for node writes (per-node mutex)
- `RwLock` for mode transitions
- Atomic counters for count/next_id

## References

1. DiskANN (NeurIPS 2019): "Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
2. FreshDiskANN (2021): "Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search"
3. HNSW (2016): "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
4. Product Quantization (2010): "Product Quantization for Nearest Neighbor Search"
5. OPQ (2013): "Optimized Product Quantization for Approximate Nearest Neighbor Search"
6. ScaNN (2020): "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
7. NSG (2019): "Fast Approximate Nearest Neighbor Search with Navigating Spreading-out Graph"
8. SSG (2019): "Satellite System Graph"
9. RaBitQ (2024): "Quantizing High-Dimensional Vectors with a Theoretical Error Bound"
10. SPANN (2021): "Highly-efficient Billion-scale Approximate Nearest Neighbor Search"

## Remaining Optimizations (TODO)

1. Visited list pool (avoid DashSet allocation per search)
2. Lazy backedge updates (only prune when neighbor is truly full)
3. Memory-mapped mode for read-only workloads
4. Custom serialization (zero-copy for vectors)
5. Graph layout optimization (BFS ordering for cache locality)
6. Parallel beam expansion
7. Optional HNSW-style multi-layer mode
