#!/bin/bash
set -euo pipefail

# Sweep learning rate and batch size for train_ts_unconditional.py
# via train_glucose_ts_unconditional.sh.
#
# Example:
#   GPU_ID=6 \
#   LR_LIST="2e-4 1e-4 5e-5" \
#   BATCH_SIZE_LIST="128 256 512" \
#   EPOCHS=80 \
#   WANDB_PROJECT=drifting-model-ts \
#   bash train_ts_unconditional_sweep_lr_bs.sh

BASE_SCRIPT="${BASE_SCRIPT:-./train_glucose_ts_unconditional.sh}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"

DATA_ROOT="${DATA_ROOT:-./AI-READI}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/sweeps/glucose_ts_unconditional_lr_bs}"
EXP_PREFIX="${EXP_PREFIX:-glucose_ts_unconditional}"

LR_LIST="${LR_LIST:-2e-4 1e-4 5e-5}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-128 256 512}"

# Shared options
NUM_WORKERS="${NUM_WORKERS:-1}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"
GLUCOSE_STRIDE="${GLUCOSE_STRIDE:-128}"
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-500}"
EVAL_METRICS="${EVAL_METRICS:-disc,vaeFID}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"
METRIC_ITERATION="${METRIC_ITERATION:-1}"
VAE_CKPT_ROOT="${VAE_CKPT_ROOT:-./fid_vae_ckpts}"
METRICS_BASE_PATH="${METRICS_BASE_PATH:-./metrics_cache}"
TS_FEATURE_ENCODER_CKPT="${TS_FEATURE_ENCODER_CKPT:-}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
EPOCHS="${EPOCHS:-}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-drifting-model-ts}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-${EXP_PREFIX}}"

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

echo "Starting TS unconditional sweep: ${total_runs} runs"
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
    EXP_NAME="${EXP_PREFIX}_${RUN_TAG}"
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
    RUN_NAME="${WANDB_RUN_PREFIX}_${RUN_TAG}"

    echo ""
    echo "============================================================"
    echo "[${run_idx}/${total_runs}] LR=${LR}, BATCH_SIZE=${BATCH_SIZE}"
    echo "EXP_NAME=${EXP_NAME}"
    echo "Output=${RUN_OUTPUT_DIR}"
    echo "W&B run=${RUN_NAME}"
    echo "============================================================"

    GPU_ID="${GPU_ID}" \
    EXP_NAME="${EXP_NAME}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
    PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SEED="${SEED}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    SAMPLE_INTERVAL="${SAMPLE_INTERVAL}" \
    GLUCOSE_STRIDE="${GLUCOSE_STRIDE}" \
    EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL}" \
    EVAL_METRICS="${EVAL_METRICS}" \
    EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES}" \
    METRIC_ITERATION="${METRIC_ITERATION}" \
    VAE_CKPT_ROOT="${VAE_CKPT_ROOT}" \
    METRICS_BASE_PATH="${METRICS_BASE_PATH}" \
    TS_FEATURE_ENCODER_CKPT="${TS_FEATURE_ENCODER_CKPT}" \
    LR="${LR}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
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
