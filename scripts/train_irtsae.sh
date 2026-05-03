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

ROOT_DIR="/playpen-shared/haochenz/Drift"
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


PARTICIPANTS_TSV="/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv"
DATA_ROOT="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed"
TRAIN_MODALITY=${1:-heart_rate}

python train_irregular_ts_ae.py \
  --data_root ${DATA_ROOT} \
  --participants_tsv_path ${PARTICIPANTS_TSV} \
  --train_modality ${TRAIN_MODALITY} \
  --use_aligned_modality \
  --train_split train \
  --val_split valid \
  \
  --anchor_modality "glucose" \
  --ts_seq_len 288 \
  --daily_min_events 288 \
  --max_anchor_gap_minutes 10 \
  --max_window_span_hours 24 \
  --anchor_sampling_minutes 5.0 \
  --anchor_sampling_tolerance_seconds 2 \
  --max_missing_ratio 0.5 \
  \
  --d_model 128 \
  --nhead 4 \
  --num_layers 3 \
  \
  --batch_size 64 \
  --epochs 500 \
  --lr 3e-4 \
  --input_random_drop_prob 0.3 \
  \
  --num_workers 1 \
  --seed 0 \
  \
  --save_dir "./outputs/irregular_ts_ae/heart_rate" \
  --save_name "best.pt" \
  \
  --wandb_project irregular-ts-ae \
  --wandb_run_name ${TRAIN_MODALITY}

