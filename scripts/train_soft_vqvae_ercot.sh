#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   DEVICE=cpu
#   TS_DATA_DIR=/Users/zhc/Documents/Time_Series_Datasets
#   SAVE_ROOT=./fid_vae_ckpts/soft_vqvae_benchmark
DEVICE=${DEVICE:-cuda}
TS_DATA_DIR=${TS_DATA_DIR:-/Users/zhc/Documents/Time_Series_Datasets}
SAVE_ROOT=${SAVE_ROOT:-./fid_vae_ckpts/soft_vqvae_benchmark}

python train_soft_vqvae_benchmark.py \
  --dataset_name ErcotData \
  --data ErcotData \
  --datasets_dir "${TS_DATA_DIR}" \
  --rel_path 'ERCOT_merged.csv' \
  --ts_seq_len 128 \
  --stride 1 \
  --delay 4 \
  --embedding 32 \
  --hidden_size 128 \
  --num_layers 2 \
  --code_dim 32 \
  --num_codes 512 \
  --ch_mult 1,2,4 \
  --temperature 0.07 \
  --kl_weight 0.01 \
  --save_dir "${SAVE_ROOT}" \
  --device "${DEVICE}"
