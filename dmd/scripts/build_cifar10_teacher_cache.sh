#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_common.sh"

TASK=${TASK:-cond}  # cond or uncond
NUM_SAMPLES=${NUM_SAMPLES:-500000}
STEPS=${STEPS:-18}
BATCH=${BATCH:-512}
SHARD_SIZE=${SHARD_SIZE:-8192}
SEED=${SEED:-0}
SIGMA_MIN=${SIGMA_MIN:-0.002}
SIGMA_MAX=${SIGMA_MAX:-80}
RHO=${RHO:-7}
DTYPE=${DTYPE:-float16}
CLASS_MODE=${CLASS_MODE:-balanced}
PREVIEW=${PREVIEW:-64}
RESUME=${RESUME:-true}

case "$TASK" in
  cond)
    MODEL=${MODEL:-$DATA_ROOT/pretrained/edm-cifar10-32x32-cond-vp.pkl}
    PER_CLASS=${PER_CLASS:-50000}
    ;;
  uncond)
    MODEL=${MODEL:-$DATA_ROOT/pretrained/edm-cifar10-32x32-uncond-vp.pkl}
    PER_CLASS=${PER_CLASS:-0}
    ;;
  *)
    echo "TASK must be 'cond' or 'uncond', got: $TASK" >&2
    exit 2
    ;;
esac

CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/edm_teacher_cache/cifar10-${TASK}-edm${STEPS}-500k}
CACHE_BUILDER=${CACHE_BUILDER:-$REPO_ROOT/dmd/scripts/build_edm_teacher_cache.py}

if [[ ! -f "$CACHE_BUILDER" ]]; then
  echo "Missing teacher cache builder: $CACHE_BUILDER" >&2
  exit 2
fi

args=(
  --teacher="$MODEL"
  --outdir="$CACHE_DIR"
  --num="$NUM_SAMPLES"
  --per-class="$PER_CLASS"
  --class-mode="$CLASS_MODE"
  --batch="$BATCH"
  --shard-size="$SHARD_SIZE"
  --seed="$SEED"
  --steps="$STEPS"
  --sigma-min="$SIGMA_MIN"
  --sigma-max="$SIGMA_MAX"
  --rho="$RHO"
  --dtype="$DTYPE"
  --preview="$PREVIEW"
)

if [[ "$RESUME" == "true" && -f "$CACHE_DIR/metadata.json" ]]; then
  args+=(--resume)
fi

echo "teacher cache: $CACHE_DIR"
"$PYTHON" "$CACHE_BUILDER" "${args[@]}"
