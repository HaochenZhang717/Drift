#!/bin/bash
set -euo pipefail

# =========================
# Basic config
# =========================

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
EXP_NAME="${EXP_NAME:-glucose_no_ts_encoder}"
DATA_ROOT="${DATA_ROOT:-./AI-READI}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXP_NAME}}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# =========================
# Runtime config
# =========================

NUM_WORKERS="${NUM_WORKERS:-1}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

# Glucose windows. train_ts_unconditional.py uses seq_len=128, delay=8,
# embedding=16 from TS_GLUCOSE_CONFIG.
GLUCOSE_STRIDE="${GLUCOSE_STRIDE:-128}"

# Set to <= 0 to disable in-training metrics.
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-500}"
EVAL_METRICS="${EVAL_METRICS:-disc,vaeFID}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"
METRIC_ITERATION="${METRIC_ITERATION:-1}"
VAE_CKPT_ROOT="${VAE_CKPT_ROOT:-./fid_vae_ckpts}"
METRICS_BASE_PATH="${METRICS_BASE_PATH:-./metrics_cache}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-drifting-model-ts}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
TS_FEATURE_ENCODER_CKPT="${TS_FEATURE_ENCODER_CKPT:-}"

echo "Starting glucose unconditional training..."
echo "Experiment: ${EXP_NAME}"
echo "GPU: ${GPU_ID}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Eval interval: ${EVAL_STEP_INTERVAL}"
echo "Eval metrics: ${EVAL_METRICS}"
echo "Eval samples: ${EVAL_NUM_SAMPLES:-full test set}"
echo "VAE ckpt root: ${VAE_CKPT_ROOT}"
echo "W&B enabled: ${WANDB_ENABLED}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run: ${WANDB_RUN_NAME}"
echo "TS feature encoder ckpt: ${TS_FEATURE_ENCODER_CKPT:-none}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${METRICS_BASE_PATH}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_ALLOC_CONF

CMD=(
python train_ts_unconditional.py
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
    --metric_iteration "${METRIC_ITERATION}" \
    --vae_ckpt_root "${VAE_CKPT_ROOT}" \
    --metrics_base_path "${METRICS_BASE_PATH}"
)

if [[ -n "${EVAL_NUM_SAMPLES}" ]]; then
    CMD+=(--eval_num_samples "${EVAL_NUM_SAMPLES}")
fi

if [[ -n "${TS_FEATURE_ENCODER_CKPT}" ]]; then
    CMD+=(--ts_feature_encoder_ckpt "${TS_FEATURE_ENCODER_CKPT}")
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
    CMD+=(
        --wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_run_name "${WANDB_RUN_NAME}"
        --wandb_mode "${WANDB_MODE}"
    )

    if [[ -n "${WANDB_ENTITY}" ]]; then
        CMD+=(--wandb_entity "${WANDB_ENTITY}")
    fi
fi

"${CMD[@]}"

echo "Training finished!"
