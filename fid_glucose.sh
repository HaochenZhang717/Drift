#!/bin/bash

# =========================
# 基础配置
# =========================

# 数据路径（改成你的）
TRAIN_PATH="./data/glucose/train_ts.npy"
VAL_PATH="./data/glucose/test_ts.npy"

# 保存目录
EXP_NAME="vae_glucose"
SAVE_DIR="./fid_vae_ckpts/${EXP_NAME}"

mkdir -p ${SAVE_DIR}

# =========================
# 训练参数
# =========================

BATCH_SIZE=128
EPOCHS=100
LR=8e-4

HIDDEN_SIZE=128
NUM_LAYERS=2
NUM_HEADS=8
LATENT_DIM=64
BETA=0.001

# =========================
# 运行
# =========================

echo "Starting training..."
echo "Experiment: ${EXP_NAME}"

python train_fid_vae.py \
    --train_path ${TRAIN_PATH} \
    --val_path ${VAL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --latent_dim ${LATENT_DIM} \
    --beta ${BETA} \
    --save_dir ${SAVE_DIR}

echo "Training finished!"