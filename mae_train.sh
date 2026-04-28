#!/bin/bash
#SBATCH --job-name=mae
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=10:00:00
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



python train_multi_scale_mae.py \
  --data_root ./AI-READI \
  --train_file glucose_train.parquet \
  --valid_file glucose_valid.parquet \
  --column glucose \
  --output_dir ./checkpoints/mae_glucose_default \
  \
  --seq_len 128 \
  --stride 32 \
  --normalize minmax \
  \
  --patch_sizes 8,16,32 \
  --embed_dim 128 \
  --latent_dim 16 \
  --encoder_depth 2 \
  --bridge_depth 1 \
  --decoder_depth 1 \
  --num_heads 4 \
  --mlp_ratio 4.0 \
  --mask_ratio 0.25 \
  --dropout 0.1 \
  --loss_weights 1.0,1.0,1.0 \
  \
  --lr 5e-4 \
  --weight_decay 1e-5 \
  --batch_size 128 \
  --epochs 500 \
  --val_interval 10 \
  \
  --device cuda:0 \
  --wandb \
  --wandb_project drifting-model-ts \
  --wandb_run_name mae_glucose_default