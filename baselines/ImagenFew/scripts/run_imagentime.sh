#!/usr/bin/env bash
#SBATCH --job-name=imagentime
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

CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

OUTPUT_ROOT="/mnt/unites8/playpen/haochenz/Drift/baselines/ImagenTime"
mkdir -p "${OUTPUT_ROOT}"

CONFIG_FILE=${1:-/playpen-shared/haochenz/Drift/baselines/ImagenFew/configs/ImagenTime/ErcotData_len256.yaml}

python /playpen-shared/haochenz/Drift/baselines/ImagenFew/run.py \
--subset_p 1.0 \
--log_dir "${OUTPUT_ROOT}" \
--config ${CONFIG_FILE}
