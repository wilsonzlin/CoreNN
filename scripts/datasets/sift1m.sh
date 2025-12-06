#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Reproducible pipeline for fetching + preparing SIFT1M assets.
# Usage:
#   scripts/datasets/sift1m.sh all            # download + convert + manifests
#   scripts/datasets/sift1m.sh download       # fetch raw data
#   scripts/datasets/sift1m.sh convert        # convert raw -> packed matrices
#   scripts/datasets/sift1m.sh manifests      # emit JSON metadata
#
# Override DATASET_ROOT to change destination (default: datasets/sift1m).
# Set SIFT_DOWNLOAD_MODE=hdf5|ftp (default: hdf5) to pick source.

ROOT=${DATASET_ROOT:-"datasets/sift1m"}
RAW="$ROOT/raw"
PROC="$ROOT/processed"
GT="$ROOT/gt"
MAN="$ROOT/manifests"
mkdir -p "$RAW" "$PROC" "$GT" "$MAN"

SIFT_BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"
SIFT_HDF5_URL="http://ann-benchmarks.com/sift-128-euclidean.hdf5"
HDF5_FILE="$RAW/sift-128-euclidean.hdf5"
DOWNLOAD_MODE=${SIFT_DOWNLOAD_MODE:-hdf5}
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

download_hdf5() {
  mkdir -p "$RAW"
  if [[ -f "$HDF5_FILE" ]]; then
    echo "[sift1m] HDF5 already present at $HDF5_FILE"
    return
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 1M -o "$HDF5_FILE" "$SIFT_HDF5_URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$SIFT_HDF5_URL" -o "$HDF5_FILE" -C -
  else
    need_tool wget
    wget -c "$SIFT_HDF5_URL" -O "$HDF5_FILE"
  fi
}

download_ftp() {
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

download() {
  case "$DOWNLOAD_MODE" in
    hdf5) download_hdf5 ;;
    ftp) download_ftp ;;
    *) echo "Unknown SIFT_DOWNLOAD_MODE=$DOWNLOAD_MODE" >&2; exit 1 ;;
  esac
}

convert_from_ftp() {
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

convert_from_hdf5() {
  need_tool python
  python - <<PY
import pathlib
try:
    import h5py
    import numpy as np
except ImportError as exc:
    raise SystemExit("h5py and numpy are required (pip install h5py numpy).") from exc

root = pathlib.Path("$ROOT")
raw = pathlib.Path("$HDF5_FILE")
proc = root / "processed"
gt = root / "gt"
proc.mkdir(parents=True, exist_ok=True)
gt.mkdir(parents=True, exist_ok=True)

chunk = 50_000
with h5py.File(raw, "r") as f:
    train = f["train"]
    test = f["test"]
    neighbors = f["neighbors"]

    def dump(dataset, dst):
        with open(dst, "wb") as out:
            for start in range(0, dataset.shape[0], chunk):
                arr = dataset[start : start + chunk].astype("<f4", copy=False)
                out.write(arr.tobytes())

    dump(train, proc / "base.f32")
    dump(test, proc / "query.f32")

    learn_count = min(100_000, train.shape[0])
    np.asarray(train[:learn_count], dtype="<f4").tofile(proc / "learn.f32")
    np.asarray(neighbors[:], dtype="<u4").tofile(gt / "sift_k100.u32")

print("Converted HDF5 -> base/query/learn/gt")
PY
}

convert() {
  if [[ "$DOWNLOAD_MODE" == "ftp" ]]; then
    convert_from_ftp
  elif [[ -f "$HDF5_FILE" ]]; then
    convert_from_hdf5
  elif [[ -f "$RAW/sift_base.fvecs" ]]; then
    convert_from_ftp
  else
    echo "[sift1m] No raw data found. Run download first." >&2
    exit 1
  fi
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
source = "$SIFT_HDF5_URL" if pathlib.Path("$HDF5_FILE").exists() else "$SIFT_BASE_URL"
assets = [
  {
    "name": "sift1m-base",
    "path": str(proc / "base.f32"),
    "source": source,
    "dim": 128,
    "dtype": "float32",
  },
  {
    "name": "sift1m-query",
    "path": str(proc / "query.f32"),
    "source": source,
    "dim": 128,
    "dtype": "float32",
  },
  {
    "name": "sift1m-learn",
    "path": str(proc / "learn.f32"),
    "source": source,
    "dim": 128,
    "dtype": "float32",
  },
  {
    "name": "sift1m-gt-k100",
    "path": str(gt / "sift_k100.u32"),
    "source": source,
    "dim": 100,
    "dtype": "uint32",
  },
]
timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for asset in assets:
    path = pathlib.Path(asset["path"])
    if not path.exists():
        print(f"warning: missing {path}; skipping manifest")
        continue
    data = path.read_bytes()
    asset["count"] = len(data) // (asset["dim"] * 4)
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
  download   Fetch raw data (default: ANN-Benchmarks HDF5 mirror).
  convert    Convert raw assets to packed matrices + uint32 gt.
  manifests  Generate JSON metadata with checksums.
  all        Run download -> convert -> manifests (default).
Env vars:
  DATASET_ROOT         Destination directory (default datasets/sift1m)
  SIFT_DOWNLOAD_MODE   hdf5 (default) or ftp
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