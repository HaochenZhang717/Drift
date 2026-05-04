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
BATCH_SIZE="${BATCH_SIZE:-256}"

# =========================
# Model / drifting config
# =========================

MODEL="${MODEL:-DriftDiT-Tiny}"
IMG_SIZE="${IMG_SIZE:-12}"
IN_CHANNELS="${IN_CHANNELS:-1}"
BATCH_N_POS="${BATCH_N_POS:-1024}"
BATCH_N_NEG="${BATCH_N_NEG:-256}"
TEMPERATURES="${TEMPERATURES:-0.02,0.05,0.2}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EPOCHS="${EPOCHS:-100}"
LOSS_DOMAIN="${LOSS_DOMAIN:-time_series}"
QUEUE_SIZE="${QUEUE_SIZE:-1280}"
USE_FEATURE_ENCODER="${USE_FEATURE_ENCODER:-0}"

# =========================
# Glucose/time-series config
# =========================
TS_SEQ_LEN="${TS_SEQ_LEN:-128}"
TS_DELAY="${TS_DELAY:-12}"
TS_EMBEDDING="${TS_EMBEDDING:-12}"
TS_STRIDE="${TS_STRIDE:-${GLUCOSE_STRIDE:-128}}"

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
echo "Model: ${MODEL}"
echo "Image size: ${IMG_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Drifting batch: n_pos=${BATCH_N_POS}, n_neg=${BATCH_N_NEG}"
echo "Temperatures: ${TEMPERATURES}"
echo "LR: ${LR}"
echo "Epochs: ${EPOCHS}"
echo "TS config: seq_len=${TS_SEQ_LEN}, delay=${TS_DELAY}, embedding=${TS_EMBEDDING}, stride=${TS_STRIDE}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${METRICS_BASE_PATH}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_ALLOC_CONF

CMD=(
    python train_ts_unconditional.py
    --output_dir "${OUTPUT_DIR}" \
    --data_root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --log_interval "${LOG_INTERVAL}" \
    --save_interval "${SAVE_INTERVAL}" \
    --sample_interval "${SAMPLE_INTERVAL}" \
    --eval_step_interval "${EVAL_STEP_INTERVAL}" \
    --eval_metrics "${EVAL_METRICS}" \
    --metric_iteration "${METRIC_ITERATION}" \
    --metrics_base_path "${METRICS_BASE_PATH}" \
    --vae_ckpt_root "${VAE_CKPT_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --model "${MODEL}" \
    --img_size "${IMG_SIZE}" \
    --in_channels "${IN_CHANNELS}" \
    --batch_n_pos "${BATCH_N_POS}" \
    --batch_n_neg "${BATCH_N_NEG}" \
    --temperatures "${TEMPERATURES}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --ema_decay "${EMA_DECAY}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --epochs "${EPOCHS}" \
    --loss_domain "${LOSS_DOMAIN}" \
    --queue_size "${QUEUE_SIZE}" \
    --ts_seq_len "${TS_SEQ_LEN}" \
    --ts_delay "${TS_DELAY}" \
    --ts_embedding "${TS_EMBEDDING}" \
    --ts_stride "${TS_STRIDE}"
)

if [[ -n "${EVAL_NUM_SAMPLES}" ]]; then
    CMD+=(--eval_num_samples "${EVAL_NUM_SAMPLES}")
fi

if [[ "${USE_FEATURE_ENCODER}" == "1" ]]; then
    CMD+=(--use_feature_encoder)
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
