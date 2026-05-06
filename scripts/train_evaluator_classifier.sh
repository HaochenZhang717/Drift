#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0


DATA_ROOT=/playpen-shared/haochenz/AI-READI-processed
PARTICIPANTS_TSV=/playpen-shared/haochenz/AI-READI-processed/participants.tsv

OUTPUT_DIR=./outputs/aireadi_eval_classifier

python train_evaluator_classifier.py \
    --data_root ${DATA_ROOT} \
    --participants_tsv_path ${PARTICIPANTS_TSV} \
    --train_split train \
    --val_split valid \
    --test_split test \
    --num_classes 4 \
    --seq_len 288 \
    --window_stride 128 \
    --window_mode daily \
    --daily_min_events 288 \
    --hidden_channels 32 \
    --dropout 0.1 \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR}