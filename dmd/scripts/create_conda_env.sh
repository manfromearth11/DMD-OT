#!/usr/bin/env bash
set -euo pipefail

# Create the CUDA/PyTorch conda environment used by the DMD-OT CIFAR scripts.
#
# Defaults are intentionally path-based so multiple servers can share the same
# repo while keeping datasets and envs under DATA_ROOT.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

DATA_ROOT=${DATA_ROOT:-/dataset/${USER}}
ENV_NAME=${ENV_NAME:-dmd-ot}
ENV_PREFIX=${ENV_PREFIX:-$DATA_ROOT/conda-envs/$ENV_NAME}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_VERSION=${TORCH_VERSION:-2.5.1+cu121}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.20.1+cu121}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. Install Miniconda/Anaconda or source conda.sh first." >&2
  exit 2
fi

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  echo "env already exists: $ENV_PREFIX"
else
  mkdir -p "$(dirname "$ENV_PREFIX")"
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
fi

PY="$ENV_PREFIX/bin/python"

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION"

"$PY" -m pip install \
  click \
  fire \
  h5py \
  imageio \
  imageio-ffmpeg \
  matplotlib \
  neptune \
  notebook \
  "numpy==1.26.4" \
  pillow \
  piq \
  psutil \
  pyspng \
  requests \
  scipy \
  tqdm \
  wandb

cat <<EOF

ready: $ENV_PREFIX

Activate:
  conda activate $ENV_PREFIX

Or run scripts directly; dmd/scripts/env_common.sh will auto-detect:
  DATA_ROOT=$DATA_ROOT dmd/scripts/prepare_cifar10_assets.sh

For manual Python commands:
  export PYTHONPATH=$REPO_ROOT/dmd:\$PYTHONPATH
EOF

