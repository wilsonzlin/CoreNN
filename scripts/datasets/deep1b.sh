#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROOT=${DATASET_ROOT:-"datasets/deep1b"}
RAW="$ROOT/raw"
PROC="$ROOT/processed"
PROC_BASE="$PROC/base"
GT="$ROOT/gt"
MAN="$ROOT/manifests"
mkdir -p "$RAW" "$PROC" "$PROC_BASE" "$GT" "$MAN"

BASE_URL="https://storage.googleapis.com/ann-datasets/deep1b"
BASE_SHARDS=(base.00.fbin base.01.fbin base.02.fbin base.03.fbin base.04.fbin base.05.fbin base.06.fbin base.07.fbin)
FILES=("${BASE_SHARDS[@]}" query.public.10K.fbin learn.350M.fbin groundtruth.public.10K.ibin)
DEEP_DIM=96
GT_K=100

need_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

fetch() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "[deep1b] skipping existing $(basename "$dest")"
    return
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 1M -o "$dest" "$url"
  else
    need_tool curl
    curl -L -C - "$url" -o "$dest"
  fi
}

download() {
  mkdir -p "$RAW"
  for f in "${FILES[@]}"; do
    fetch "$BASE_URL/$f" "$RAW/$f"
  done
}

convert_fbin() {
  local src="$1"
  local dst="$2"
  local expected_dim="$3"
  python - "$src" "$dst" "$expected_dim" <<'PY'
import pathlib, shutil, struct, sys
src, dst, expected_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(src, "rb") as f_in:
    dim = struct.unpack('<I', f_in.read(4))[0]
    count = struct.unpack('<I', f_in.read(4))[0]
    if expected_dim and dim != expected_dim:
        raise SystemExit(f"dimension mismatch: expected {expected_dim}, got {dim}")
    with open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
meta_path = pathlib.Path(dst + ".meta.json")
meta_path.write_text(
    '{"dim": %d, "count": %d, "source": "%s"}\n' % (dim, count, src)
)
print(f"Converted {src} -> {dst} ({count} x {dim})")
PY
}

convert_ibin() {
  local src="$1"
  local dst="$2"
  local expected_dim="$3"
  python - "$src" "$dst" "$expected_dim" <<'PY'
import json, numpy as np, pathlib, struct, sys
src, dst, expected_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(src, "rb") as f:
    dim = struct.unpack('<I', f.read(4))[0]
    count = struct.unpack('<I', f.read(4))[0]
    if expected_dim and dim != expected_dim:
        raise SystemExit(f"dimension mismatch: expected {expected_dim}, got {dim}")
    data = np.frombuffer(f.read(), dtype=np.int32).reshape(count, dim).astype(np.uint32)
with open(dst, "wb") as out:
    out.write(data.tobytes())
meta = {"dim": dim, "count": count, "source": src}
pathlib.Path(dst + ".meta.json").write_text(json.dumps(meta) + "\n")
print(f"Converted {src} -> {dst} ({count} x {dim})")
PY
}

convert() {
  need_tool python
  for shard in "${BASE_SHARDS[@]}"; do
    local stem=${shard%.fbin}
    convert_fbin "$RAW/$shard" "$PROC_BASE/${stem}.f32" "$DEEP_DIM"
  done
  convert_fbin "$RAW/query.public.10K.fbin" "$PROC/query_10k.f32" "$DEEP_DIM"
  convert_fbin "$RAW/learn.350M.fbin" "$PROC/learn_350m.f32" "$DEEP_DIM"
  convert_ibin "$RAW/groundtruth.public.10K.ibin" "$GT/deep1b_k100.u32" "$GT_K"
}

manifests() {
  need_tool python
  python - <<PY
import hashlib, json, pathlib
root = pathlib.Path("$ROOT")
man = root / "manifests"
man.mkdir(parents=True, exist_ok=True)
assets = []
proc_base = root / "processed" / "base"
for shard in sorted(proc_base.glob("base.*.f32")):
    assets.append({
        "name": shard.stem,
        "path": str(shard),
        "dim": $DEEP_DIM,
        "dtype": "float32",
        "source": f"$BASE_URL/{shard.stem}.fbin",
    })
extra = [
    {"name": "deep1b-query", "path": str(root / "processed" / "query_10k.f32"), "dim": $DEEP_DIM, "dtype": "float32", "source": "$BASE_URL/query.public.10K.fbin"},
    {"name": "deep1b-learn", "path": str(root / "processed" / "learn_350m.f32"), "dim": $DEEP_DIM, "dtype": "float32", "source": "$BASE_URL/learn.350M.fbin"},
    {"name": "deep1b-gt-k100", "path": str(root / "gt" / "deep1b_k100.u32"), "dim": $GT_K, "dtype": "uint32", "source": "$BASE_URL/groundtruth.public.10K.ibin"},
]
assets.extend(extra)
for asset in assets:
    path = pathlib.Path(asset["path"])
    if not path.exists():
        print(f"warning: missing {path}; skipping manifest")
        continue
    data = path.read_bytes()
    asset["size_bytes"] = len(data)
    bytes_per_row = 4 * asset["dim"]
    asset["count"] = len(data) // bytes_per_row
    asset["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path = man / f"{asset['name']}.json"
    manifest_path.write_text(json.dumps(asset, indent=2) + "\\n")
    print(f"wrote {manifest_path}")
PY
}

usage() {
  cat <<'EOF'
Usage: scripts/datasets/deep1b.sh [download|convert|manifests|all]
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
