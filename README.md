# Roxanne

Production-scale database for [querying **billions** of vectors and embeddings](https://en.wikipedia.org/wiki/Nearest_neighbor_search) in sublinear time on commodity machines.

With Roxanne, you can:
- Build an index over 1 billion embeddings in under 1 hour using **GPUs or TPUs**.
- Serve the index from a machine with only **64 GB RAM**.
- Perform **real-time updates and deletes** in-place without rebuilding or rewriting.
- Achieve **thousands of QPS** and **90+% recall**.

Read the [accompanying blog post](https://blog.wilsonl.in/roxanne) for an accessible introduction and in-depth technical report.

## Why use?

**Designed with a novel disk-first algorithm and data layout**

Roxanne is designed to use cheap plentiful flash storage, not expensive DRAM, retaining high accuracy and throughput while costing 40x–100x less. This means you can index more vectors than can fit in memory.

It supports true in-place updates and deletes such that they free space and don't degrade query performance, and don't require a full rebuild or dump.

**Unified self-contained disk format across broad tools**

Roxanne uses a shared core library with a consistent data storage format, so you can use it across your favorite stack, interact with whatever tool is the most handy, and have it adapt with your growing infra and codebase. One example journey:

1. Use the **CLI** to import an existing dataset and use Roxanne in scripts.
1. Start building a **Python** prototype app locally, using roxanne-py.
1. **Rclone** the database folder to a production datacenter VM.
1. Run the Roxanne **server** to provide an HTTP REST API in production.

**Lots of nice goodies**

- Supports FP16 so you can store double the amount of vectors in memory and on disk with little accuracy loss.
- Queries don't block updates and vice versa.
- Import directly from an existing HNSW index with no rebuilding.
- Seamlessly scales from 1 to 1 billion embeddings without downtime, dynamically adjusting strategy automatically.
- &hellip; and more
