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


GPU_ID=0 \
EXP_NAME=glucose_no_ts_encoder \
NUM_WORKERS=4 \
USE_FEATURE_ENCODER=0 \
WANDB_RUN_NAME="data_loss_0.1,0.2" \
TEMPERATURES=0.1,0.2 \
EPOCHS=100 \
bash train_glucose_ts_unconditional.sh

GPU_ID=0 \
EXP_NAME=glucose_no_ts_encoder \
NUM_WORKERS=4 \
USE_FEATURE_ENCODER=0 \
WANDB_RUN_NAME="data_loss_0.05,0.2" \
TEMPERATURES=0.05,0.2 \
EPOCHS=100 \
bash train_glucose_ts_unconditional.sh


