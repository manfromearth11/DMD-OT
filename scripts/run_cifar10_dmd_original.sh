#!/usr/bin/env bash
set -euo pipefail

# Canonical CIFAR-10 DMD baseline launcher.
# Override any variable at call time, e.g.:
#   BATCH_SIZE=64 LAMBDA_K=0 scripts/run_cifar10_dmd_original.sh

export TASK=${TASK:-cond}
export METHOD=paired
export BATCH_SIZE=${BATCH_SIZE:-128}
export LAMBDA_K=${LAMBDA_K:-1}
export DMD_SAMPLE_MODE=${DMD_SAMPLE_MODE:-free}
export DMD_LOSS_LAMBDA=${DMD_LOSS_LAMBDA:-0.25}
export MAX_STEPS=${MAX_STEPS:-50000}
export EVAL_EVERY=${EVAL_EVERY:-10000}
export FID_NUM_SAMPLES=${FID_NUM_SAMPLES:-50000}
export WANDB_PROJECT=${WANDB_PROJECT:-cifar10-dmd-original}
export NAME=${NAME:-${TASK}_dmd_original_b${BATCH_SIZE}_kl${LAMBDA_K}_${MAX_STEPS}}

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dmd/scripts/train_cifar10_cached_dmd.sh"
