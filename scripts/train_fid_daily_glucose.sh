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


DATA_ROOT=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed
PARTICIPANTS_TSV=/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv

python train_fid_vae_daily_glucose.py \
  --data_root ${DATA_ROOT} \
  --participants_tsv_path ${PARTICIPANTS_TSV} \
  --batch_size 128 \
  --epochs 100 \
  --lr 8e-4 \
  --hidden_size 128 \
  --num_layers 2 \
  --num_heads 8 \
  --latent_dim 64 \
  --beta 0.001
