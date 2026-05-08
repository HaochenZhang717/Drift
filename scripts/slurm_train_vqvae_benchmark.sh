#!/usr/bin/env bash
#SBATCH --job-name=vqvae-bench
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

ROOT_DIR="/playpen-shared/haochenz/Drift"
cd "$ROOT_DIR"

mkdir -p /playpen-shared/haochenz/logs/slurm

source ~/.zshrc >/dev/null 2>&1 || true

CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm

echo "Running from: $ROOT_DIR"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

bash scripts/train_vqvae_benchmark.sh
