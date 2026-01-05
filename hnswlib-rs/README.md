# hnswlib-rs

Pure-Rust HNSW (Hierarchical Navigable Small World) graph for approximate nearest-neighbor search, inspired by the original C++ `hnswlib`.

This crate intentionally decouples the **graph** from **vector storage**:
- `Hnsw<K, M>` owns the graph + a mapping from your external key `K` to an internal dense `NodeId`.
- You provide a `VectorStore` keyed by `NodeId` to supply vectors on demand.

Vector types:
- Dense floating point: `f32`, `f16`, `bf16` (distance computation accumulates in `f32`).
- Per-vector quantized int8: `Qi8Ref { data: &[i8], scale: f32, zero_point: i8 }`.

## Quickstart

```rust
use hnswlib_rs::{Hnsw, HnswConfig, InMemoryVectorStore, L2, Result};

fn main() -> Result<()> {
  let dim = 128;
  let max_nodes = 100_000;

  let cfg = HnswConfig::new(dim, max_nodes)
    .m(16)
    .ef_construction(200)
    .ef_search(50);

  let hnsw = Hnsw::new(L2::new(), cfg);
  let vectors = InMemoryVectorStore::<f32>::new(dim, max_nodes);

  let v = vec![0.0; dim];
  hnsw.insert(&vectors, "doc-1".to_string(), &v)?;

  let hits = hnsw.search(&vectors, &v, 10, None)?;
  assert_eq!(hits[0].key, "doc-1");
  Ok(())
}
```

## Core concepts

- **`NodeId`**: dense internal id allocated by the graph (stable for the lifetime of the index).
- **`VectorStore`**: your vector backend keyed by `NodeId` (can return borrowed slices or owned buffers).
- **`Metric`**: distance function (e.g. `L2`, `Cosine`, `InnerProduct`).

If you want to fetch a vector by your external key, do:
1) `let id = hnsw.node_id(&key)?;`
2) `let v = vectors.vector(id).ok_or(Error::MissingVector)?;`

## Why `NodeId`?

HNSW’s hot path is graph traversal: iterating neighbor lists, tracking a visited set, and updating per-node link lists.
Using your external key type `K` directly in those internals would force expensive and/or bulky representations (hashing, cloning, larger neighbor entries, non-dense visited/lock structures).

`NodeId` exists to keep the core graph representation **dense, fast, and easy to make correct**:
- Neighbor lists are compact (stored as `u32` IDs internally).
- Per-node state is stored in contiguous arrays (levels, tombstones, locks, visited tags, linklists).
- Your `VectorStore` can be implemented efficiently with dense storage (e.g. `Vec` indexed by `NodeId`), while still letting you keep vectors elsewhere if you want.
- The legacy `hnswlib` format already uses dense internal IDs, so loading maps naturally onto `NodeId`.

`NodeId`s are **not reused** for different keys: `delete` tombstones the node; `set` updates/resurrects the same `NodeId`. Reuse would invalidate stable `NodeId` handles held by a `VectorStore` (and by callers).

## Mutation semantics

- `insert(key, vector)`: fails if `key` already exists.
- `set(key, vector)`: insert-or-update; if the key was deleted, it is resurrected and connections are repaired.
- `delete(key)`: tombstones the node (keeps the key mapping; use `set` to resurrect).

## Concurrency

`Hnsw` is designed for concurrent search + concurrent mutation.

The provided `InMemoryVectorStore` supports lock-free reads and parallel updates (per-`NodeId` atomic swap).

## Persistence

Use `Hnsw::save_to()` / `Hnsw::load_from()` to save/load the **graph + key mapping + config** via `std::io::Write` / `std::io::Read` (sequential `bincode`).

Notes:
- Vectors are **not** included; persist your `VectorStore` separately (keyed by `NodeId`).
- The metric/space is not stored; you must provide the `Metric` when loading.
- The graph file includes `dim` and `dtype`; `load_from` validates `dtype` against the `Metric`’s vector type.

```rust
use hnswlib_rs::{Hnsw, HnswConfig, InMemoryVectorStore, L2, Result};

fn save_and_load() -> Result<()> {
  let dim = 128;
  let max_nodes = 100_000;

  let hnsw = Hnsw::new(L2::new(), HnswConfig::new(dim, max_nodes));
  let vectors = InMemoryVectorStore::<f32>::new(dim, max_nodes);
  hnsw.insert(&vectors, "doc-1".to_string(), &vec![0.0; dim])?;

  let mut f = std::fs::File::create("hnsw.bin")?;
  hnsw.save_to(&mut f)?;

  let mut f = std::fs::File::open("hnsw.bin")?;
  let loaded = Hnsw::load_from(L2::new(), &mut f)?;
  assert_eq!(loaded.len(), hnsw.len());
  Ok(())
}
```

## Persisting vectors (`InMemoryVectorStore`)

`InMemoryVectorStore` provides `save_to` / `load_from` for a dense matrix keyed by `NodeId` order.

The on-disk format includes a small `bincode` header (`dtype`, `dim`, `max_nodes`, `node_count`), followed by raw row-major scalar bytes (little-endian).

```rust
use hnswlib_rs::{Hnsw, HnswConfig, InMemoryVectorStore, L2, Result};

fn save_and_load() -> Result<()> {
  let dim = 128;
  let max_nodes = 100_000;

  let hnsw = Hnsw::new(L2::new(), HnswConfig::new(dim, max_nodes));
  let store = InMemoryVectorStore::<f32>::new(dim, max_nodes);
  hnsw.insert(&store, "doc-1".to_string(), &vec![0.0; dim])?;
  let node_count = hnsw.len();

  let mut f = std::fs::File::create("vectors.bin")?;
  store.save_to(&mut f, node_count)?;

  let mut f = std::fs::File::open("vectors.bin")?;
  let (loaded, loaded_count) = InMemoryVectorStore::<f32>::load_from(&mut f)?;
  assert_eq!(loaded_count, node_count);
  Ok(())
}
```

## Per-vector QI8 (quantized int8) vectors

Use `InMemoryQi8VectorStore` with `L2Qi8`, `CosineQi8`, or `InnerProductQi8`.

```rust
use hnswlib_rs::{Hnsw, HnswConfig, InMemoryQi8VectorStore, L2Qi8, Qi8Ref, Result};

fn qi8_example() -> Result<()> {
  let dim = 128;
  let max_nodes = 100_000;

  let hnsw = Hnsw::new(L2Qi8::new(), HnswConfig::new(dim, max_nodes));
  let store = InMemoryQi8VectorStore::new(dim, max_nodes);

  let v = vec![0i8; dim];
  let q = Qi8Ref { data: &v, scale: 0.02, zero_point: 0 };
  hnsw.insert(&store, 1u64, q)?;

  let hits = hnsw.search(&store, q, 10, None)?;
  assert_eq!(hits[0].key, 1u64);
  Ok(())
}
```

## Legacy `hnswlib` loader (read-only)

`legacy::load_hnswlib` loads the original C++ `hnswlib` on-disk format:

```rust
use hnswlib_rs::{legacy::load_hnswlib, L2, VectorStore};

let bytes = std::fs::read("index.bin")?;
let (graph, vectors) = load_hnswlib(L2::new(), 128, &bytes)?;

let label: u64 = 123;
let id = graph.node_id(&label)?;
let v = vectors.vector(id).unwrap();
```

Notes:
- The legacy format does not store the metric/space name; you must provide a `Metric`.
- The loader is zero-copy over `&[u8]`.
- For zero-copy `f32` casting, the input bytes must be aligned for `f32` (mmap’d files are fine).

## Non-goals

- API compatibility with the C++ `hnswlib` API.
- Writing the legacy `hnswlib` format (loading is supported).
