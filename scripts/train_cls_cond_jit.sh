#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="/playpen-shared/haochenz/Drift"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/cls_cond_jit_run1"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

DATA_ROOT="/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed"
PARTICIPANTS_TSV_PATH="/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv"

python train_cls_cond_jit.py \
    --data_root "${PROJECT_ROOT}/AI-READI" \
    --participants_tsv_path "" \
    --train_split train \
    --val_split valid \
    --output_dir "${OUTPUT_DIR}" \
    \
    --epochs 50 \
    --batch_size 32 \
    --num_workers 4 \
    --seed 42 \
    \
    --lr 1e-4 \
    --weight_decay 5e-2 \
    --grad_clip 1.0 \
    --log_interval 20 \
    --save_interval 10 \
    \
    --img_size 18 \
    --patch_size 2 \
    --in_channels 1 \
    --hidden_channels 256 \
    --depth 6 \
    --num_heads 4 \
    --attn_dropout 0.1 \
    --proj_dropout 0.1 \
    --bottleneck_dim 64 \
    \
    --label_drop_prob 0.0 \
    --P_mean -1.2 \
    --P_std 1.2 \
    --t_eps 1e5 \
    --noise_scale 1.0 \
    --ema_decay1 0.999 \
    --ema_decay2 0.9999 \
    \
    --sampling_method heun \
    --num_sampling_steps 50 \
    --cfg 1.0 \
    --interval_min 0.0 \
    --interval_max 1.0 \
    \
    --ts_seq_len 288 \
    --ts_delay 1 \
    --ts_embedding ${IMG_SIZE} \
    --window_mode daily \
    --daily_min_events 288 \
    --max_anchor_gap_minutes 30 \
    --max_window_span_hours 24 \
    --anchor_sampling_minutes 5 \
    --anchor_sampling_tolerance_seconds 120 \
    --log_interval 20 \
    --save_interval 10 \
    --wandb \
    --wandb_project drifting-model \
    --wandb_run_name cls_cond_jit_run1