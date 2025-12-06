# Dataset + Benchmark Pipeline

Living manual for producing reproducible ANN benchmarks on CoreNN. Covers dataset acquisition (SIFT1M + Deep1B), preprocessing, ground-truth generation, and evaluation harness conventions. Keep this document updated as new datasets or tooling land.

## 1. Goals
- Produce identical binary artifacts (vectors, queries, ground truth) across machines.
- Automate ingestion + evaluation so we can compare commits by SHA.
- Support both developer-scale (SIFT1M) and production-scale (Deep1B) corpora using the same layout.

## 2. Directory Layout
All datasets live under the workspace root:

```
datasets/
  sift1m/
    raw/                # untouched downloads
    processed/          # packed float32/float16 matrices
    gt/                 # ground-truth neighbor ids
    manifests/          # JSON metadata per artifact
  deep1b/
    ... (same layout)
```

Outputs from benchmark runs go to `docs/benchmarks/YYYY-MM-DD/` (see `performance_master_plan.md`).

## 3. Tooling Requirements
- Python ≥ 3.10 with `numpy`, `torch` (CUDA/ROCm builds), `cupy`, `cuml`, `msgpack`, `tqdm`.
- Sufficient GPU memory (>= 12 GB recommended for Deep1B ground-truth batches; lower VRAM works with smaller `--batch-size`).
- System packages: `wget`, `curl`, `aria2c`, `pigz`, `tar`.
- Rust toolchain (nightly per `rust-toolchain.toml`).

Suggested virtualenv setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install cupy-cuda12x cuml-cuda12x msgpack tqdm
```
Adjust CUDA/ROCm wheels per GPU vendor.

## 4. SIFT1M Pipeline (Reference-Scale)
Fast path (one command, idempotent):
```bash
scripts/datasets/sift1m.sh all
```
Manual breakdown (mirrors what the script does):
1. **Download TEXMEX fvecs/ivecs**
   ```bash
   mkdir -p datasets/sift1m/raw
   cd datasets/sift1m/raw
   wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift_base.fvecs
   wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift_query.fvecs
   wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift_groundtruth.ivecs
   wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift_learn.fvecs
   ```

2. **Convert to packed matrices (float32 rows)**
   ```bash
   cd /workspace
   python tools/convert_texmex.py datasets/sift1m/raw/sift_base.fvecs --dtype float32 --out datasets/sift1m/processed/base.f32
   python tools/convert_texmex.py datasets/sift1m/raw/sift_query.fvecs --dtype float32 --out datasets/sift1m/processed/query.f32
   python tools/convert_texmex.py datasets/sift1m/raw/sift_learn.fvecs --dtype float32 --out datasets/sift1m/processed/learn.f32
   ```
   Ground truth already provided (`sift_groundtruth.ivecs`). Convert to uint32 packed matrix:
   ```bash
   python - <<'PY'
import numpy as np, struct
src = "datasets/sift1m/raw/sift_groundtruth.ivecs"
with open(src, "rb") as f: raw = f.read()
dim = int.from_bytes(raw[:4], "little")
assert dim == 100  # SIFT1M GT k = 100
vecs = np.vstack([
    np.frombuffer(raw, dtype=np.int32, count=dim, offset=(4+4*dim)*i + 4)
    for i in range(len(raw)//(4+4*dim))
]).astype(np.uint32)
with open("datasets/sift1m/gt/sift_k100.u32", "wb") as f: f.write(vecs.tobytes())
PY
   ```

3. **Optional subsampling for rapid iteration**
   ```bash
   python tools/sample_vecs.py datasets/sift1m/processed/base.f32 --dim 128 --n 100000 --out datasets/sift1m/processed/base_100k.f32
   python tools/sample_vecs.py datasets/sift1m/processed/query.f32 --dim 128 --n 1000 --out datasets/sift1m/processed/query_1k.f32
   ```

4. **Generate custom ground truth (if k or metric changes)**
   ```bash
   python tools/calc_nn.py \
     --vectors datasets/sift1m/processed/base.f32 \
     --dim 128 \
     --n-samples 10000 \
     --batch-size 512 \
     --k 100 \
     --out-queries datasets/sift1m/processed/query_10k.f32 \
     --out-results datasets/sift1m/gt/sift_k100_custom.u32
   ```

5. **Manifest metadata**
   Create `datasets/sift1m/manifests/base.json` describing SHA256, dim, dtype, source URL, creation command. Use this to prove reproducibility.

6. **Baseline evaluation**
   ```bash
   cargo run -p corenn --release -- eval \
     datasets/sift1m/corenn.db \
     --vectors datasets/sift1m/processed/base.f32 \
     --dim 128 \
     --queries datasets/sift1m/processed/query_10k.f32 \
     --results datasets/sift1m/gt/sift_k100_custom.u32 \
     --k 100 \
     --dtype f32
   ```
   Capture metrics + config in `docs/benchmarks/<date>/sift1m.json`.

## 5. Deep1B Pipeline (Scale Testing)
Fast path (streams ~400 GB over the network; ensure you have disk space first):
```bash
scripts/datasets/deep1b.sh all
```
Manual breakdown:
Deep1B is hosted at [https://github.com/arbabench/benchmarks/tree/main/data](https://github.com/arbabench/benchmarks/tree/main/data) and mirrors of `http://ann-benchmarks.com/deep1b`.

1. **Download base + learn + query shards**
   ```bash
   mkdir -p datasets/deep1b/raw
   cd datasets/deep1b/raw
   aria2c -x 16 https://storage.googleapis.com/ann-datasets/deep1b/base.00.fbin
   # repeat for base.01.fbin … base.07.fbin (8 shards)
   aria2c -x 16 https://storage.googleapis.com/ann-datasets/deep1b/query.public.10K.fbin
   aria2c -x 16 https://storage.googleapis.com/ann-datasets/deep1b/learn.350M.fbin
   aria2c -x 16 https://storage.googleapis.com/ann-datasets/deep1b/groundtruth.public.10K.ibin
   ```

2. **Concatenate shards**
   ```bash
   cat base.*.fbin > datasets/deep1b/raw/base_all.fbin
   ```

3. **Convert FBIN -> packed matrices**
   FBIN format stores (uint32 dim, uint32 count, then floats). Use Python helper:
   ```bash
   python - <<'PY'
import numpy as np, struct
import pathlib
from typing import Tuple

def read_fbin(src: str) -> np.ndarray:
    with open(src, "rb") as f:
        dim = struct.unpack('I', f.read(4))[0]
        n = struct.unpack('I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        assert data.size == n * dim
        return data.reshape(n, dim)

def convert(src, dst):
    arr = read_fbin(src)
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'wb') as f: f.write(arr.tobytes())

convert("datasets/deep1b/raw/base_all.fbin", "datasets/deep1b/processed/base.f32")
convert("datasets/deep1b/raw/query.public.10K.fbin", "datasets/deep1b/processed/query_10k.f32")
convert("datasets/deep1b/raw/learn.350M.fbin", "datasets/deep1b/processed/learn_350m.f32")
PY
   ```

4. **Ground truth**
   Convert `groundtruth.public.10K.ibin` (uint32 neighbors) using similar helper, storing `datasets/deep1b/gt/deep1b_k100.u32`.

5. **Optional downsampling**
   For bring-up/testing, sample 50M-100M vectors using `tools/sample_vecs.py` (with `--n` and `--dim 96`).

6. **Custom ground truth for new query sets**
   ```bash
   python tools/calc_nn.py \
     --vectors datasets/deep1b/processed/base.f32 \
     --dim 96 \
     --n-samples 20000 \
     --batch-size 256 \
     --k 100 \
     --out-queries datasets/deep1b/processed/query_20k.f32 \
     --out-results datasets/deep1b/gt/deep1b_k100_custom.u32
   ```
   Expect long runtimes; monitor GPU thermals.

7. **Baseline evaluation**
   ```bash
   cargo run -p corenn --release -- eval \
     datasets/deep1b/corenn.db \
     --vectors datasets/deep1b/processed/base_chunk.f32 \
     --dim 96 \
     --queries datasets/deep1b/processed/query_20k.f32 \
     --results datasets/deep1b/gt/deep1b_k100_custom.u32 \
     --k 100 \
     --dtype f16
   ```
   - Use chunked ingestion (100k–1M vectors per batch) to avoid long single transactions.
   - Document RocksDB path + disk type.

## 6. Automation Hooks
- `scripts/datasets/sift1m.sh` and `scripts/datasets/deep1b.sh` encapsulate download/convert/manifest steps (see headers for usage).
- Next: create `Makefile` targets (`make data-sift1m`, `make data-deep1b`) that delegate to those scripts and store logs in `docs/benchmarks`.
- Extend CI to validate checksums (lightweight) and optionally run a smoke benchmark on SIFT1M nightly.

## 7. Metadata & Manifests
Each processed artifact should have a JSON entry:
```json
{
  "name": "sift1m-base",
  "source": "ftp://ftp.irisa.fr/local/texmex/corpus/sift_base.fvecs",
  "dim": 128,
  "dtype": "float32",
  "count": 1000000,
  "sha256": "...",
  "generated_by": "python tools/convert_texmex.py ...",
  "created_at": "2025-12-05T01:05:00Z",
  "notes": "Full base set, no subsampling"
}
```
Store these in `datasets/<name>/manifests/*.json` and reference them in benchmark logs.

## 8. Next Steps
- Wire the dataset scripts into Makefile targets + CI caching for team-friendly reuse.
- Integrate dataset download + preprocessing into GitHub Actions cache to avoid repeated large transfers.
- Add verification tests ensuring processed matrix dimensions/dtypes match manifest metadata.
