#!/bin/bash
set -euo pipefail

# Grid sweep for learning rate and batch size, based on:
#   train_full_series_ts2vec_glucose.sh
#
# Example:
#   GPU_ID=6 \
#   LR_LIST="1e-3 5e-4 1e-4" \
#   BATCH_SIZE_LIST="256 512 1024" \
#   EPOCHS=600 \
#   WANDB_PROJECT=drifting-model-ts \
#   bash train_full_series_ts2vec_glucose_sweep_lr_bs.sh

BASE_SCRIPT="${BASE_SCRIPT:-./train_full_series_ts2vec_glucose.sh}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"

DATA_ROOT="${DATA_ROOT:-./AI-READI}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./feature_extractors/checkpoints/sweeps/full_series_ts2vec_glucose_lr_bs}"

# Space-separated lists.
LR_LIST="${LR_LIST:-1e-3 5e-4 1e-4}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-256 512 1024}"

# Shared run options (can be overridden from env).
SEQ_LEN="${SEQ_LEN:-128}"
STRIDE="${STRIDE:-32}"
EPOCHS="${EPOCHS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.2}"
HIDDEN_DIMS="${HIDDEN_DIMS:-64}"
OUTPUT_DIMS="${OUTPUT_DIMS:-128}"
DEPTH="${DEPTH:-10}"
MASK_PROB="${MASK_PROB:-0.85}"
CROP_MIN_RATIO="${CROP_MIN_RATIO:-1.0}"
CROP_MAX_RATIO="${CROP_MAX_RATIO:-1.0}"
VAL_REPEATS="${VAL_REPEATS:-1}"
SAVE_FULL_SERIES_FEATURES="${SAVE_FULL_SERIES_FEATURES:-0}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-drifting-model-ts}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-full_series_ts2vec_glucose}"

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base script not found: ${BASE_SCRIPT}"
  exit 1
fi

run_idx=0
total_runs=0
for _lr in ${LR_LIST}; do
  for _bs in ${BATCH_SIZE_LIST}; do
    total_runs=$((total_runs + 1))
  done
done

echo "Starting sweep: ${total_runs} runs"
echo "LR_LIST=${LR_LIST}"
echo "BATCH_SIZE_LIST=${BATCH_SIZE_LIST}"
echo "GPU=${GPU_ID}"

for LR in ${LR_LIST}; do
  for BATCH_SIZE in ${BATCH_SIZE_LIST}; do
    run_idx=$((run_idx + 1))

    lr_tag="${LR//./p}"
    lr_tag="${lr_tag//-/_m_}"
    bs_tag="${BATCH_SIZE}"
    RUN_TAG="lr_${lr_tag}__bs_${bs_tag}"
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
    RUN_NAME="${WANDB_RUN_PREFIX}_${RUN_TAG}"

    echo ""
    echo "============================================================"
    echo "[${run_idx}/${total_runs}] LR=${LR}, BATCH_SIZE=${BATCH_SIZE}"
    echo "Output: ${RUN_OUTPUT_DIR}"
    echo "W&B run: ${RUN_NAME}"
    echo "============================================================"

    GPU_ID="${GPU_ID}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
    SEQ_LEN="${SEQ_LEN}" \
    STRIDE="${STRIDE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
    LR="${LR}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    TEMPERATURE="${TEMPERATURE}" \
    HIDDEN_DIMS="${HIDDEN_DIMS}" \
    OUTPUT_DIMS="${OUTPUT_DIMS}" \
    DEPTH="${DEPTH}" \
    MASK_PROB="${MASK_PROB}" \
    CROP_MIN_RATIO="${CROP_MIN_RATIO}" \
    CROP_MAX_RATIO="${CROP_MAX_RATIO}" \
    VAL_REPEATS="${VAL_REPEATS}" \
    SAVE_FULL_SERIES_FEATURES="${SAVE_FULL_SERIES_FEATURES}" \
    WANDB_ENABLED="${WANDB_ENABLED}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_RUN_NAME="${RUN_NAME}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_MODE="${WANDB_MODE}" \
    bash "${BASE_SCRIPT}"
  done
done

echo ""
echo "Sweep finished: ${total_runs} runs completed."
