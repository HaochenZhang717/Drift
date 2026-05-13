#python benchmarking_latent_drift.py \
#  --output_dir ./outputs/latent_drift \
#  --model DriftDiT-Tiny \
#  --batch_size 128 \
#  --batch_n_pos 256 \
#  --batch_n_neg 256 \
#  --temperatures 0.02,0.05,0.2 \
#  --epochs 1000 \
#  --queue_size 2048 \
#  --ts_seq_len 512 \
#  --ts_delay 8 \
#  --ts_embedding 64 \
#  --ts_stride 1 \
#  --one_channel \
#  --decode_average_overlap \
#  --softvq_ckpt_path /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/ErcotData/best.pt \
#  --dataset_name ErcotData \
#  --data ErcotData \
#  --datasets_dir /mnt/unites8/haochenz/Time_Series_Datasets \
#  --rel_path ERCOT_merged.csv


#python benchmarking_latent_drift.py \
#  --output_dir ./outputs/latent_drift \
#  --model DriftDiT-Tiny \
#  --batch_size 128 \
#  --batch_n_pos 256 \
#  --batch_n_neg 256 \
#  --temperatures 0.02,0.05,0.2 \
#  --epochs 1000 \
#  --queue_size 2048 \
#  --ts_seq_len 512 \
#  --ts_delay 8 \
#  --ts_embedding 64 \
#  --ts_stride 1 \
#  --one_channel \
#  --decode_average_overlap \
#  --softvq_ckpt_path /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/ErcotData/best.pt \
#  --dataset_name ErcotData \
#  --data ErcotData \
#  --datasets_dir /mnt/unites8/haochenz/Time_Series_Datasets \
#  --rel_path ERCOT_merged.csv



#!/usr/bin/env bash
set -euo pipefail

# ===== paths =====
ROOT_DIR="/playpen-shared/haochenz/Drift"
cd "${ROOT_DIR}"
# 你训练好的 soft-vqvae checkpoint（按需改）
SOFTVQ_CKPT="/mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/GlucoseSliding/best.pt"
# Glucose 数据目录（按你机器实际路径改）
GLUCOSE_DATA_DIR="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed"
# 输出目录
OUT_DIR="/mnt/unites8/playpen/haochenz/Drift/latent_drift_glucose"
# ===== run =====
python benchmarking_latent_drift.py \
  --output_dir "${OUT_DIR}" \
  --seed 42 \
  --num_workers 4 \
  --log_interval 50 \
  --sample_interval 10 \
  --batch_size 512 \
  --model DriftDiT-Tiny \
  --batch_n_pos 1024 \
  --batch_n_neg 512 \
  --temperatures 0.02,0.05,0.2 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --ema_decays 0.999,0.995,0.99 \
  --warmup_steps 1000 \
  --epochs 50 \
  --queue_size 2048 \
  --ts_seq_len 512 \
  --ts_delay 8 \
  --ts_embedding 64 \
  --ts_stride 128 \
  --decode_average_overlap \
  --softvq_ckpt_path "${SOFTVQ_CKPT}" \
  --dataset_name GlucoseSliding \
  --data GlucoseSliding \
  --datasets_dir "${GLUCOSE_DATA_DIR}" \
  --rel_path 'glucose_{split}.parquet'
