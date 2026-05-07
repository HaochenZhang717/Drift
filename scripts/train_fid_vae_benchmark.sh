#!/bin/bash
set -euo pipefail

# One Channel
#python train_fid_vae_benchmark.py \
#  --dataset_name "ETTm1" \
#  --data "ETTm1" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/ETT-small/ETTm1.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#  --one_channel
#
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "ETTm2" \
#  --data "ETTm2" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/ETT-small/ETTm2.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#  --one_channel
#
#
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "ETTh2" \
#  --data "ETTh2" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/ETT-small/ETTh2.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#  --one_channel
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "Weather" \
#  --data "custom" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/weather/weather.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#  --one_channel
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "AirQuality" \
#  --data "AirQuality" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSG/AirQuality/AirQualityUCI.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#  --one_channel
#
## Multi Channel
#python train_fid_vae_benchmark.py \
#  --dataset_name "ETTm1" \
#  --data "ETTm1" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/ETT-small/ETTm1.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "ETTm2" \
#  --data "ETTm2" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/ETT-small/ETTm2.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#
#
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "ETTh2" \
#  --data "ETTh2" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/ETT-small/ETTh2.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#
#
#python train_fid_vae_benchmark.py \
#  --dataset_name "Weather" \
#  --data "custom" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSF/weather/weather.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \
#


#python train_fid_vae_benchmark.py \
#  --dataset_name "AirQuality" \
#  --data "AirQuality" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "TSG/AirQuality/AirQualityUCI.csv" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \



#python train_fid_vae_benchmark.py \
#  --dataset_name "mujoco" \
#  --data "mujoco" \
#  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
#  --rel_path "./TSG/mujoco0.0" \
#  --ts_seq_len 256 \
#  --batch_size 128 \
#  --epochs 100 \
#  --save_dir ./fid_vae_ckpts/benchmark_256 \


TS_LENGTHS=(128 256 512 64)
#TS_LENGTHS=(64)

for TSLEN in "${TS_LENGTHS[@]}"; do

  echo "========================================"
  echo "Running ERCOT with TSLEN=${TSLEN}"
  echo "========================================"

  python train_fid_vae_benchmark.py \
    --dataset_name "ErcotData" \
    --data "ErcotData" \
    --datasets_dir "/mnt/unites8/playpen/haochenz/Time_Series_Datasets" \
    --rel_path "ERCOT_merged.csv" \
    --ts_seq_len ${TSLEN} \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --weight_decay 1e-3 \
    \
    --hidden_size 32 \
    --num_layers 1 \
    --latent_dim 4 \
    --latent_downsample 16 \
    --decoder_upsample_rate 4 \
    --dropout 0.1 \
    --beta 0.01 \
    \
    --save_dir /mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_ercot_${TSLEN} \
    --wandb \
    --wandb_project "FID_VAE" \
    --wandb_run_name "ErcotData_len${TSLEN}"

  echo "========================================"
  echo "Running Household with TSLEN=${TSLEN}"
  echo "========================================"

  python train_fid_vae_benchmark.py \
    --dataset_name "HouseholdData" \
    --data "HouseholdData" \
    --datasets_dir "/mnt/unites8/playpen/haochenz/Time_Series_Datasets" \
    --rel_path "HouseHold_6.csv" \
    --ts_seq_len ${TSLEN} \
    --window_stride 10 \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --weight_decay 1e-3 \
    \
    --hidden_size 32 \
    --num_layers 1 \
    --latent_dim 8 \
    --latent_downsample 16 \
    --decoder_upsample_rate 4 \
    --dropout 0.1 \
    --beta 0.01 \
    \
    --save_dir /mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_household_${TSLEN} \
    --wandb \
    --wandb_project "FID_VAE" \
    --wandb_run_name "HouseholdData_len${TSLEN}"

done
