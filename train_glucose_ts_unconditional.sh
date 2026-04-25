#!/bin/bash
set -euo pipefail

# =========================
# Basic config
# =========================

GPU_ID="${GPU_ID:-0}"
EXP_NAME="${EXP_NAME:-glucose_ts_unconditional}"
DATA_ROOT="${DATA_ROOT:-./AI-READI}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXP_NAME}}"

# =========================
# Runtime config
# =========================

NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

# Glucose windows. train_ts_unconditional.py uses seq_len=128, delay=8,
# embedding=16 from TS_GLUCOSE_CONFIG.
GLUCOSE_STRIDE="${GLUCOSE_STRIDE:-128}"

# Set to a positive number to enable in-training metrics.
# Keeping this off by default avoids expensive metric jobs during a first run.
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-0}"
EVAL_METRICS="${EVAL_METRICS:-disc}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-1000}"
METRIC_ITERATION="${METRIC_ITERATION:-1}"

echo "Starting glucose unconditional training..."
echo "Experiment: ${EXP_NAME}"
echo "GPU: ${GPU_ID}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_ts_unconditional.py \
    --dataset glucose \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --log_interval "${LOG_INTERVAL}" \
    --save_interval "${SAVE_INTERVAL}" \
    --sample_interval "${SAMPLE_INTERVAL}" \
    --glucose_stride "${GLUCOSE_STRIDE}" \
    --eval_step_interval "${EVAL_STEP_INTERVAL}" \
    --eval_metrics "${EVAL_METRICS}" \
    --eval_num_samples "${EVAL_NUM_SAMPLES}" \
    --metric_iteration "${METRIC_ITERATION}"

echo "Training finished!"
