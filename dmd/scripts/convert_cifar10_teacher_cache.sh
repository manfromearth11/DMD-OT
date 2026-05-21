#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_common.sh"

TASK=${TASK:-cond}  # cond or uncond
STEPS=${STEPS:-18}
CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/edm_teacher_cache/cifar10-${TASK}-edm${STEPS}-500k}
OUTPUT=${OUTPUT:-$DATA_ROOT/dmd_data/cifar10-${TASK}-edm${STEPS}-500k-array.hdf5}
MAX_SAMPLES=${MAX_SAMPLES:-}

args=(
  --cache-dir="$CACHE_DIR"
  --output="$OUTPUT"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples="$MAX_SAMPLES")
fi

echo "convert cache: $CACHE_DIR"
echo "output h5:     $OUTPUT"
"$PYTHON" "$REPO_ROOT/dmd/scripts/edm_cache_to_h5.py" "${args[@]}"

