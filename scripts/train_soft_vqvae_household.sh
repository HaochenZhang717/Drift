#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   DEVICE=cpu
#   TS_DATA_DIR=/Users/zhc/Documents/Time_Series_Datasets
#   SAVE_ROOT=./fid_vae_ckpts/soft_vqvae_benchmark
#   TS_SEQ_LEN=128
#   DELAY=4
#   EMBEDDING=32
DEVICE=${DEVICE:-cuda}
TS_DATA_DIR=${TS_DATA_DIR:-/Users/zhc/Documents/Time_Series_Datasets}
SAVE_ROOT=${SAVE_ROOT:-./fid_vae_ckpts/soft_vqvae_benchmark}
TS_SEQ_LEN=${TS_SEQ_LEN:-128}
DELAY=${DELAY:-4}
EMBEDDING=${EMBEDDING:-32}

python train_soft_vqvae_benchmark.py \
  --dataset_name HouseholdData \
  --data HouseholdData \
  --datasets_dir "${TS_DATA_DIR}" \
  --rel_path 'HouseHold_6.csv' \
  --ts_seq_len "${TS_SEQ_LEN}" \
  --stride 10 \
  --delay "${DELAY}" \
  --embedding "${EMBEDDING}" \
  --hidden_size 32 \
  --num_layers 2 \
  --code_dim 8 \
  --num_codes 512 \
  --ch_mult 1,2,2,4 \
  --temperature 0.07 \
  --kl_weight 0.01 \
  --save_dir "${SAVE_ROOT}" \
  --device "${DEVICE}"
