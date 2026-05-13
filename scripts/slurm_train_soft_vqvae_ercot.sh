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
CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO


DEVICE=${DEVICE:-cuda}
TS_DATA_DIR=${TS_DATA_DIR:-/Users/zhc/Documents/Time_Series_Datasets}
SAVE_ROOT=${SAVE_ROOT:-./fid_vae_ckpts/soft_vqvae_benchmark}
TS_SEQ_LEN=${TS_SEQ_LEN:-128}
DELAY=${DELAY:-4}
EMBEDDING=${EMBEDDING:-32}
TEMPERATURE=${TEMPERATURE:-0.07}
NUM_CODES=${NUM_CODES:-512}
KL_WEIGHT=${KL_WEIGHT:-0.01}

python train_soft_vqvae_benchmark.py \
  --dataset_name ErcotData \
  --data ErcotData \
  --datasets_dir "${TS_DATA_DIR}" \
  --rel_path 'ERCOT_merged.csv' \
  --ts_seq_len "${TS_SEQ_LEN}" \
  --stride 1 \
  --delay "${DELAY}" \
  --embedding "${EMBEDDING}" \
  --hidden_size 32 \
  --num_layers 2 \
  --code_dim 8 \
  --num_codes "${NUM_CODES}" \
  --ch_mult 1,2,2,4 \
  --temperature "${TEMPERATURE}" \
  --kl_weight "${KL_WEIGHT}" \
  --save_dir "${SAVE_ROOT}" \
  --device "${DEVICE}"
