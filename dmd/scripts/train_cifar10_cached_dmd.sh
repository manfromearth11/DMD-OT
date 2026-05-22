#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_common.sh"

TASK=${TASK:-cond}          # cond or uncond
METHOD=${METHOD:-paired}    # paired or ot
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-10000}
FID_NUM_SAMPLES=${FID_NUM_SAMPLES:-50000}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LAMBDA_K=${LAMBDA_K:-0}
DMD_SAMPLE_MODE=${DMD_SAMPLE_MODE:-free}
OT_EPS=${OT_EPS:-0.2}
OT_ITERS=${OT_ITERS:-30}
NUM_WORKERS=${NUM_WORKERS:-0}
CACHE_DATA=${CACHE_DATA:-true}
LOG_WANDB=${LOG_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-cifar10-dmd-sanity-test}
SEED=${SEED:-42}
PRINT_STEPS=${PRINT_STEPS:-100}
IM_SAVE_STEPS=${IM_SAVE_STEPS:-10000}
EPOCHS=${EPOCHS:-1000000}
STEPS=${STEPS:-18}
OUTROOT=${OUTROOT:-$DATA_ROOT/dmd_ot_runs/cifar10_cached_dmd}
FID_REF=${FID_REF:-$DATA_ROOT/pretrained/cifar10-32x32.npz}

fire_bool() {
  case "$1" in
    true|True|TRUE|1|yes|Yes|YES) echo True ;;
    false|False|FALSE|0|no|No|NO) echo False ;;
    *)
      echo "Expected boolean-like value, got: $1" >&2
      exit 2
      ;;
  esac
}

case "$TASK" in
  cond)
    MODEL=${MODEL:-$DATA_ROOT/pretrained/edm-cifar10-32x32-cond-vp.pkl}
    DATA_PATH=${DATA_PATH:-$DATA_ROOT/dmd_data/cifar10-cond-edm${STEPS}-500k-array.hdf5}
    CLASS_BATCH=${CLASS_BATCH:-true}
    DMD_LOSS_LAMBDA=${DMD_LOSS_LAMBDA:-0.25}
    ;;
  uncond)
    MODEL=${MODEL:-$DATA_ROOT/pretrained/edm-cifar10-32x32-uncond-vp.pkl}
    DATA_PATH=${DATA_PATH:-$DATA_ROOT/dmd_data/cifar10-uncond-edm${STEPS}-500k-array.hdf5}
    CLASS_BATCH=${CLASS_BATCH:-false}
    DMD_LOSS_LAMBDA=${DMD_LOSS_LAMBDA:-0.5}
    ;;
  *)
    echo "TASK must be 'cond' or 'uncond', got: $TASK" >&2
    exit 2
    ;;
esac

case "$METHOD" in
  paired|ot) ;;
  *)
    echo "METHOD must be 'paired' or 'ot', got: $METHOD" >&2
    exit 2
    ;;
esac

if [[ "${TRAIN_DIFFUSER:-auto}" == "auto" ]]; then
  if [[ "$LAMBDA_K" == "0" || "$LAMBDA_K" == "0.0" || "$LAMBDA_K" == "0.00" ]]; then
    TRAIN_DIFFUSER=false
  else
    TRAIN_DIFFUSER=true
  fi
fi

TRAIN_DIFFUSER_ARG=$(fire_bool "$TRAIN_DIFFUSER")
CACHE_DATA_ARG=$(fire_bool "$CACHE_DATA")
LOG_WANDB_ARG=$(fire_bool "$LOG_WANDB")
CLASS_BATCH_ARG=$(fire_bool "$CLASS_BATCH")

NAME=${NAME:-${TASK}_${METHOD}_b${BATCH_SIZE}_kl${LAMBDA_K}_eps${OT_EPS}_${MAX_STEPS}_$(date +%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$OUTROOT/$NAME}
mkdir -p "$RUN_DIR"

args=(
  --epochs="$EPOCHS"
  --batch-size="$BATCH_SIZE"
  --eval-batch-size="$EVAL_BATCH_SIZE"
  --num-workers="$NUM_WORKERS"
  --max-steps="$MAX_STEPS"
  --eval-every-steps="$EVAL_EVERY"
  --fid-ref-path="$FID_REF"
  --fid-num-samples="$FID_NUM_SAMPLES"
  --lambda-k="$LAMBDA_K"
  --dmd-sample-mode="$DMD_SAMPLE_MODE"
  --dmd-loss-lambda="$DMD_LOSS_LAMBDA"
  --lr="$LR"
  --weight-decay="$WEIGHT_DECAY"
  --train-diffuser="$TRAIN_DIFFUSER_ARG"
  --cache-data="$CACHE_DATA_ARG"
  --im-save-steps="$IM_SAVE_STEPS"
  --print-steps="$PRINT_STEPS"
  --seed="$SEED"
  --log-wandb="$LOG_WANDB_ARG"
  --wandb-project="$WANDB_PROJECT"
  --wandb-name="$NAME"
  --model-path="$MODEL"
  --teacher-model-path="$MODEL"
  --data-path="$DATA_PATH"
  --output-dir="$RUN_DIR"
  --class-batch="$CLASS_BATCH_ARG"
)

if [[ "$METHOD" == "ot" ]]; then
  args+=(
    --use-ot-reg=True
    --ot-eps="$OT_EPS"
    --ot-iters="$OT_ITERS"
  )
fi

if [[ "$(fire_bool "${SKIP_FID:-false}")" == "True" ]]; then
  args+=(--skip-fid=True)
fi

echo "run:       $NAME"
echo "task:      $TASK"
echo "method:    $METHOD"
echo "batch:     $BATCH_SIZE"
echo "lambda_k:  $LAMBDA_K"
echo "dmd_sample:$DMD_SAMPLE_MODE"
echo "classbatch:$CLASS_BATCH"
echo "data:      $DATA_PATH"
echo "out:       $RUN_DIR"

"$PYTHON" -m dmd train "${args[@]}"
