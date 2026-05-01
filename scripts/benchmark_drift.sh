#!/usr/bin/env bash
#SBATCH --job-name=drift
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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

# =========================
# 基本设置
# =========================
#DATASETS_DIR=/Users/zhc/Documents/PhD/projects/ImagenFew/data
DATASETS_DIR="/playpen-shared/haochenz/ImagenFew/data"

#REL_PATH=TSF/ETT-small/ETTm1.csv
#DATA_BACKEND=ETTm1
#DATASET_NAME=ETTm1
#IN_CHANNEL=7

REL_PATH=${REL_PATH:-TSF/ETT-small/ETTm1.csv}
DATA_BACKEND=${DATA_BACKEND:-ETTm1}
DATASET_NAME=${DATASET_NAME:-ETTm1}
IN_CHANNEL=${IN_CHANNEL-7}

#PROJECT_ROOT="/Users/zhc/Documents/PhD/projects/drifting-model"
PROJECT_ROOT="/playpen-shared/haochenz/Drift"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${DATA_BACKEND}"
VAE_ROOT="${PROJECT_ROOT}/fid_vae_ckpts"


# =========================
# 训练参数
# =========================
BATCH_SIZE=4
BS_POS=4
EPOCHS=200
IMG_SIZE=16
TS_LEN=256

# =========================
# 运行
# =========================
python benchmarking_drift.py \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num_workers 16 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    \
    --model DriftDiT-Tiny \
    --img_size ${IMG_SIZE} \
    --in_channels ${IN_CHANNEL} \
    \
    --batch_n_pos ${BS_POS} \
    --batch_n_neg ${BATCH_SIZE} \
    --temperatures 0.02,0.05,0.2 \
    \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    \
    --ema_decay 0.999 \
    --warmup_steps 1000 \
    \
    --loss_domain time_series \
    --queue_size 1280 \
    \
    --ts_seq_len ${TS_LEN} \
    --ts_delay ${IMG_SIZE} \
    --ts_embedding ${IMG_SIZE} \
    --ts_stride 1 \
    \
    --dataset_name "${DATASET_NAME}" \
    --data "${DATA_BACKEND}" \
    --datasets_dir "${DATASETS_DIR}" \
    --rel_path "${REL_PATH}" \
    \
    --eval_metrics vaeFID \
    --eval_num_samples 2000 \
    --eval_step_interval 500 \
    \
    --vae_ckpt_root "${VAE_ROOT}" \
    \
    --wandb \
    --wandb_project drifting-model-ts \
    --wandb_run_name glucose_uncond
