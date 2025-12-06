#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Reproducible pipeline for fetching + preparing SIFT1M assets.
# Usage:
#   scripts/datasets/sift1m.sh all            # download + convert + manifests
#   scripts/datasets/sift1m.sh download       # only fetch raw fvecs/ivecs
#   scripts/datasets/sift1m.sh convert        # convert raw -> packed matrices
#   scripts/datasets/sift1m.sh manifests      # emit JSON metadata
#
# Override DATASET_ROOT to change destination (default: datasets/sift1m).

ROOT=${DATASET_ROOT:-"datasets/sift1m"}
RAW="$ROOT/raw"
PROC="$ROOT/processed"
GT="$ROOT/gt"
MAN="$ROOT/manifests"
mkdir -p "$RAW" "$PROC" "$GT" "$MAN"

SIFT_BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"
FILES=(
  sift_base.fvecs
  sift_query.fvecs
  sift_learn.fvecs
  sift_groundtruth.ivecs
)

need_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

download() {
  need_tool wget
  pushd "$RAW" >/dev/null
  for f in "${FILES[@]}"; do
    if [[ -f $f ]]; then
      echo "[sift1m] skipping existing $f"
      continue
    fi
    wget -c "$SIFT_BASE_URL/$f"
  done
  popd >/dev/null
}

convert() {
  need_tool python
  pushd "$REPO_ROOT" >/dev/null
  python tools/convert_texmex.py "$RAW/sift_base.fvecs" --dtype float32 --out "$PROC/base.f32"
  python tools/convert_texmex.py "$RAW/sift_query.fvecs" --dtype float32 --out "$PROC/query.f32"
  python tools/convert_texmex.py "$RAW/sift_learn.fvecs" --dtype float32 --out "$PROC/learn.f32"
  popd >/dev/null
  python - <<PY
import numpy as np, pathlib
root = pathlib.Path("$GT")
root.mkdir(parents=True, exist_ok=True)
src = pathlib.Path("$RAW") / "sift_groundtruth.ivecs"
with open(src, "rb") as f:
    raw = f.read()
dim = int.from_bytes(raw[:4], "little")
count = len(raw) // (4 + 4 * dim)
rows = []
for i in range(count):
    offset = (4 + 4 * dim) * i + 4
    rows.append(np.frombuffer(raw, dtype=np.int32, count=dim, offset=offset))
mat = np.vstack(rows).astype(np.uint32)
with open(root / "sift_k100.u32", "wb") as f:
    f.write(mat.tobytes())
print(f"Wrote {mat.shape} ground-truth matrix")
PY
}

manifests() {
  need_tool python
  python - <<PY
import hashlib, json, pathlib, time
root = pathlib.Path("$ROOT")
proc = root / "processed"
gt = root / "gt"
manifest_dir = root / "manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)
assets = [
  {
    "name": "sift1m-base",
    "path": str(proc / "base.f32"),
    "source": "$SIFT_BASE_URL/sift_base.fvecs",
    "dim": 128,
    "dtype": "float32",
    "count": 1_000_000,
  },
  {
    "name": "sift1m-query",
    "path": str(proc / "query.f32"),
    "source": "$SIFT_BASE_URL/sift_query.fvecs",
    "dim": 128,
    "dtype": "float32",
    "count": 10_000,
  },
  {
    "name": "sift1m-learn",
    "path": str(proc / "learn.f32"),
    "source": "$SIFT_BASE_URL/sift_learn.fvecs",
    "dim": 128,
    "dtype": "float32",
    "count": 100_000,
  },
  {
    "name": "sift1m-gt-k100",
    "path": str(gt / "sift_k100.u32"),
    "source": "$SIFT_BASE_URL/sift_groundtruth.ivecs",
    "dim": 100,
    "dtype": "uint32",
    "count": 10_000,
  },
]
timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for asset in assets:
    path = pathlib.Path(asset["path"])
    if not path.exists():
        print(f"warning: missing {path}; skipping manifest")
        continue
    data = path.read_bytes()
    asset["sha256"] = hashlib.sha256(data).hexdigest()
    asset["size_bytes"] = len(data)
    asset["created_at"] = timestamp
    manifest_path = manifest_dir / f"{asset['name']}.json"
    manifest_path.write_text(json.dumps(asset, indent=2) + "\\n")
    print(f\"wrote {manifest_path}\")
PY
}

usage() {
  cat <<'EOF'
Usage: scripts/datasets/sift1m.sh [download|convert|manifests|all]
  download   Fetch raw TEXMEX files via wget.
  convert    Convert raw assets to packed matrices + uint32 gt.
  manifests  Generate JSON metadata with checksums.
  all        Run download -> convert -> manifests (default).
EOF
}

cmd=${1:-all}
case "$cmd" in
  download) download ;;
  convert) convert ;;
  manifests) manifests ;;
  all) download; convert; manifests ;;
  *) usage; exit 1 ;;
esac
***