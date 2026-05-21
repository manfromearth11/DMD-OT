#!/usr/bin/env bash

# Shared local environment for DMD-OT scripts. Source this file from wrappers.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "env_common.sh is meant to be sourced, not executed." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export DATA_ROOT=${DATA_ROOT:-/dataset/${USER}}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-$TMPDIR/pycache}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$WANDB_DIR/cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$WANDB_DIR/config}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  DMD_ENV_NAME=${DMD_ENV_NAME:-dmd-ot}
  python_candidates=(
    "$DATA_ROOT/conda-envs/$DMD_ENV_NAME/bin/python"
    "$DATA_ROOT/conda-envs/edm-h100-lpips/bin/python"
  )
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    python_candidates+=("$CONDA_PREFIX/bin/python")
  fi
  for candidate in "${python_candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      export PYTHON="$candidate"
      break
    fi
  done
  export PYTHON=${PYTHON:-python}
fi

export PYTHONPATH="$REPO_ROOT/dmd:${PYTHONPATH:-}"

for libdir in \
  /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib \
  /usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib \
  /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib
do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done
