#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_common.sh"

CHECKPOINT=${CHECKPOINT:-}
FID_REF=${FID_REF:-$DATA_ROOT/pretrained/cifar10-32x32.npz}
NUM=${NUM:-50000}
BATCH=${BATCH:-128}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/last_checkpoint.pt" >&2
  exit 2
fi

OUT=${OUT:-$(dirname "$CHECKPOINT")/corrected_fid_${NUM}_seed${SEED}.json}

"$PYTHON" "$REPO_ROOT/dmd/scripts/eval_checkpoint_fid.py" \
  --checkpoint="$CHECKPOINT" \
  --ref="$FID_REF" \
  --num="$NUM" \
  --batch="$BATCH" \
  --seed="$SEED" \
  --device="$DEVICE" \
  --out="$OUT"

