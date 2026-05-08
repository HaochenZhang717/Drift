#!/bin/bash
set -euo pipefail

TS_LENGTHS=(128 256 512 64)

for TSLEN in "${TS_LENGTHS[@]}"; do

  echo "========================================"
  echo "Running ERCOT with TSLEN=${TSLEN}"
  echo "========================================"

  python train_vqvae_benchmark.py \
    --dataset_name "ErcotData" \
    --data "ErcotData" \
    --datasets_dir "/mnt/unites8/playpen/haochenz/Time_Series_Datasets" \
    --rel_path "ERCOT_merged.csv" \
    --ts_seq_len ${TSLEN} \
    --window_stride 1 \
    --ts_stride 1 \
    --stride 1 \
    --train_split "train" \
    --val_split "test" \
    \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --weight_decay 1e-3 \
    \
    --hidden_size 32 \
    --num_layers 1 \
    --code_dim 4 \
    --num_codes 1000 \
    --latent_downsample 16 \
    --decoder_upsample_rate 4 \
    --dropout 0.1 \
    --commitment_weight 0.25 \
    \
    --save_dir /mnt/unites8/playpen/haochenz/Drift/vqvae_ckpts/benchmark_ercot_${TSLEN} \
    --device "cuda" \
    --wandb \
    --wandb_project "VQVAE" \
    --wandb_run_name "ErcotData_len${TSLEN}"

  echo "========================================"
  echo "Running Household with TSLEN=${TSLEN}"
  echo "========================================"

  python train_vqvae_benchmark.py \
    --dataset_name "HouseholdData" \
    --data "HouseholdData" \
    --datasets_dir "/mnt/unites8/playpen/haochenz/Time_Series_Datasets" \
    --rel_path "HouseHold_6.csv" \
    --ts_seq_len ${TSLEN} \
    --window_stride 10 \
    --ts_stride 10 \
    --stride 10 \
    --train_split "train" \
    --val_split "test" \
    \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --weight_decay 1e-3 \
    \
    --hidden_size 32 \
    --num_layers 1 \
    --code_dim 8 \
    --num_codes 1000 \
    --latent_downsample 16 \
    --decoder_upsample_rate 4 \
    --dropout 0.1 \
    --commitment_weight 0.25 \
    \
    --save_dir /mnt/unites8/playpen/haochenz/Drift/vqvae_ckpts/benchmark_household_${TSLEN} \
    --device "cuda" \
    --wandb \
    --wandb_project "VQVAE" \
    --wandb_run_name "HouseholdData_len${TSLEN}"

done
