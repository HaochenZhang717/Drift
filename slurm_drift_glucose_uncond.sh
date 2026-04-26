#!/usr/bin/env bash
#SBATCH --job-name=drift
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  else
    echo "Could not find a usable conda binary." >&2
    exit 1
  fi
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

#!/bin/bash
set -euo pipefail

# =========================
# Basic config
# =========================

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
EXP_NAME="${EXP_NAME:-glucose_ts_unconditional}"
DATA_ROOT="${DATA_ROOT:-./AI-READI}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXP_NAME}}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

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
