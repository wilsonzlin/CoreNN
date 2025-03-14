# Roxanne

Production-scale database for [querying **billions** of vectors and embeddings](https://en.wikipedia.org/wiki/Nearest_neighbor_search) in sublinear time on commodity machines.

With Roxanne, you can:
- Build a high-accuracy index over 1 billion embeddings in under 1 hour cost-effectively using **GPUs or TPUs**.
- Serve the 1B index from a machine with only **64 GB RAM** — that's more vectors than memory.
- Perform **real-time updates and deletes** in-place without rebuilding or rewriting.
- Achieve **thousands of QPS** and **95+% recall**.

Read the [accompanying blog post](https://blog.wilsonl.in/roxanne) for an accessible introduction and in-depth technical report.

## Performance

Roxanne achieves **top marks in all quadrants**: accuracy, build speed, build cost, query performance, and scalability.

## Why use Roxanne?

**Disk-first algorithm and data layout**

Roxanne is designed from the ground up to use flash storage, not expensive DRAM, retaining high accuracy and throughput while costing 40x–100x less. This means you can index more vectors than can fit in memory. It's derived from a [new algorithm](https://suhasjs.github.io/files/diskann_neurips19.pdf) designed to exploit SSD characteristics, where current algorithms fail due to being designed for RAM access patterns.

**Real-time updates and deletes**

Roxanne supports true in-place updates and deletes such that they free space and don't degrade query performance, and don't require a full rebuild or dump. Other vector databases and algorithms operate by soft-deleting and building small immutable shards, which degrade query performance and accuracy.

Roxanne can handle thousands of inserts, updates, and deletes per second. Queries don't block updates and vice versa. It seamlessly scales from 1 to 1 billion vectors without downtime, dynamically adjusting strategy automatically.

**Unified experience**

Roxanne uses a shared core library with a consistent self-contained data storage format, so you can use it across your favorite stack, interact with whatever tool is the most handy, and have it adapt with your growing infra and codebase. One example journey:

1. Use the **CLI** to import an existing dataset and use Roxanne in scripts.
1. Start building a **Python** prototype app locally, using roxanne-py.
1. **Rclone** the database folder to a production datacenter VM.
1. Run the Roxanne **server** to provide an HTTP REST API in production.
1. Use Roxanne as an **embedded** database in a Rust program with roxanne-rs.

**Lots of nice goodies**

- Supports FP16 so you can store double the amount of vectors in memory and on disk.
- Import directly from an existing HNSW index with no rebuilding.
- Comes in a single self-contained static binary.
- Batteries-included toolkit for doing evals, migrations, imports/exports, quantization, and more.
- Written in Rust, backed by RocksDB.
- &hellip; and more

## Getting started
