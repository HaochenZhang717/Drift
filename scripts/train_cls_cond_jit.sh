#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2


EXP_NAME="cls_cond_jit_run1"
OUTPUT_DIR="/mnt/unites8/playpen/haochenz/Drift/outputs/${EXP_NAME}"

PROJECT_ROOT="/playpen-shared/haochenz/Drift"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

DATA_ROOT="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed"
PARTICIPANTS_TSV_PATH="/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv"
IMG_SIZE=18

python train_cls_cond_jit.py \
    --data_root "${DATA_ROOT}" \
    --participants_tsv_path "${PARTICIPANTS_TSV_PATH}" \
    --train_split train \
    --val_split valid \
    --output_dir "${OUTPUT_DIR}" \
    \
    --epochs 1000 \
    --batch_size 256 \
    --num_workers 4 \
    --seed 42 \
    \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 100 \
    \
    --img_size ${IMG_SIZE} \
    --patch_size 2 \
    --in_channels 1 \
    --hidden_channels 128 \
    --depth 4 \
    --num_heads 4 \
    --attn_dropout 0.1 \
    --proj_dropout 0.1 \
    --bottleneck_dim 64 \
    --num_classes 4 \
    \
    --label_drop_prob 0.0 \
    --P_mean -1.2 \
    --P_std 1.2 \
    --t_eps 1e-5 \
    --noise_scale 1.0 \
    \
    --ema_decay1 0.999 \
    --ema_decay2 0.9999 \
    \
    --sampling_method heun \
    --num_sampling_steps 50 \
    --cfg 1.0 \
    \
    --interval_min 0.0 \
    --interval_max 1.0 \
    \
    --ts_seq_len 288 \
    --ts_delay ${IMG_SIZE} \
    --ts_embedding ${IMG_SIZE} \
    \
    --window_mode daily \
    --daily_min_events 288 \
    \
    --max_missing_ratio 0.5 \
    \
    --max_anchor_gap_minutes 10 \
    --max_window_span_hours 24 \
    --anchor_sampling_minutes 5 \
    --anchor_sampling_tolerance_seconds 2 \
    \
    --wandb \
    --wandb_project class_cgm \
    --wandb_run_name ${EXP_NAME}
