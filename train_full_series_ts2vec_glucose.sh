#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
DATA_ROOT="${DATA_ROOT:-./AI-READI}"
OUTPUT_DIR="${OUTPUT_DIR:-./feature_extractors/checkpoints/full_series_ts2vec_glucose}"

SEQ_LEN="${SEQ_LEN:-128}"
STRIDE="${STRIDE:-32}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.2}"

# Model size.
HIDDEN_DIMS="${HIDDEN_DIMS:-64}"
OUTPUT_DIMS="${OUTPUT_DIMS:-128}"
DEPTH="${DEPTH:-10}"

# Probability of keeping each timestamp. MASK_PROB=0.8 means mask 20%.
MASK_PROB="${MASK_PROB:-0.85}"

# Keep both at 1.0 for full-window views. Set e.g. 0.5/1.0 to add crop augmentation.
CROP_MIN_RATIO="${CROP_MIN_RATIO:-1.0}"
CROP_MAX_RATIO="${CROP_MAX_RATIO:-1.0}"

MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-}"
MAX_VALID_WINDOWS="${MAX_VALID_WINDOWS:-4096}"
VAL_REPEATS="${VAL_REPEATS:-1}"
SAVE_FULL_SERIES_FEATURES="${SAVE_FULL_SERIES_FEATURES:-0}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-drifting-model-ts}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-full_series_ts2vec_glucose}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CMD=(
python train_full_series_ts2vec_glucose.py
  --device cuda
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --seq_len "${SEQ_LEN}"
  --stride "${STRIDE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --hidden_dims "${HIDDEN_DIMS}"
  --output_dims "${OUTPUT_DIMS}"
  --depth "${DEPTH}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --temperature "${TEMPERATURE}"
  --mask_prob "${MASK_PROB}"
  --crop_min_ratio "${CROP_MIN_RATIO}"
  --crop_max_ratio "${CROP_MAX_RATIO}"
  --max_valid_windows "${MAX_VALID_WINDOWS}"
  --val_repeats "${VAL_REPEATS}"
)

if [[ -n "${MAX_TRAIN_WINDOWS}" ]]; then
  CMD+=(--max_train_windows "${MAX_TRAIN_WINDOWS}")
fi

if [[ "${SAVE_FULL_SERIES_FEATURES}" == "1" ]]; then
  CMD+=(--save_full_series_features)
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

echo "Training full-series TS2Vec glucose encoder..."
echo "GPU: ${GPU_ID}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Mask keep prob: ${MASK_PROB}"
echo "Crop ratio: ${CROP_MIN_RATIO}-${CROP_MAX_RATIO}"
echo "Model: hidden_dims=${HIDDEN_DIMS}, output_dims=${OUTPUT_DIMS}, depth=${DEPTH}"
echo "W&B enabled: ${WANDB_ENABLED}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run: ${WANDB_RUN_NAME}"

"${CMD[@]}"
