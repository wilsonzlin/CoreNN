# CoreNN Performance Scratchpad

*Fast-moving notebook for calculations, hypotheses, and running tasks. Feel free to reorganize frequently; nothing here is sacred. Promote durable insights into `performance_master_plan.md`.*

## Working Log
| Timestamp (UTC) | Activity | Notes / Findings |
| --- | --- | --- |
| 2025-12-05T00:00 | Session start | Set up comprehensive plan + scratchpad.
| 2025-12-05T00:45 | Repo audit | No historical benchmark data or metrics in tree.
| 2025-12-05T01:15 | Dataset pipeline | Authored `docs/dataset_pipeline.md` for SIFT1M & Deep1B.
| 2025-12-05T01:35 | Automation | Added `scripts/datasets/sift1m.sh` downloader + converter.
| 2025-12-05T02:05 | Automation | Added `scripts/datasets/deep1b.sh` downloader + converter.
| 2025-12-05T02:20 | Data fetch | `sift1m.sh download` failed (FTP PASV timeout / missing file).

## Immediate Next Actions
- [x] Catalog existing benchmarks / gather any historical numbers. *(None checked in.)*
- [x] Stand up reproducible dataset pipeline (SIFT1M + Deep1B) using `tools/` scripts. *(Documented steps; automation pending.)*
- [ ] Run baseline `corenn cmd eval` to capture recall + latency.
- [ ] Profile query hot path with `cargo flamegraph -p corenn --bin corenn -- eval ...`.
- [x] Script Deep1B dataset automation.
- [ ] Identify reliable SIFT1M mirrors (FTP unreliable).

## Notes / Hypotheses
- Search path currently single-layer; likely to benefit from multiple entry points or coarse quantizers.
- Compression threshold (`cfg.compression_threshold`) defaults to 10M; may delay PQ activation for evaluation datasets—consider manual override.
- `query_search_list_cap` defaults to `2 * max_edges (128)`; SOTA systems typically use 200-400 for higher recall.

## Open Questions
1. What hardware targets (CPU, RAM, storage) must we optimize for first?
2. Are there existing latency/recall SLAs from stakeholders?
3. Can we assume float16 ingestion universally, or do we need per-tenant dtype mixes?
4. Do we need to support incremental PQ retraining without downtime?

## Parking Lot / Ideas
- Evaluate using `tantivy`/`quickwit`-style inverted index for hybrid filtering before ANN search.
- Investigate `mmap`-backed vector slabs to reduce copy overhead between RocksDB and compute layer.
- Consider hooking into `tokio-metrics` to expose search stats via server endpoint.

## Links / References To Check Later
- [ ] DiskANN & FreshDiskANN implementation details on block layout.
- [ ] Vamana DAG optimization in Google ScaNN.
- [ ] RAFT/cuVS GPU distance kernels.

## Misc Calculations
_Add quick math here (e.g., PQ codebook sizes, memory budgets)._