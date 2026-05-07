#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=2

DATA_ROOT=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed
PARTICIPANTS_TSV=/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv

# Conditional generator checkpoint from cls_cond_no_cfg training
GEN_CKPT=/playpen-shared/haochenz/Drift/outputs/glucose_cls_cond_no_cfg/aireadi_study_group_glucose/checkpoint_final.pt

# Trained evaluator classifier checkpoint
CLF_CKPT=/playpen-shared/haochenz/Drift/outputs/aireadi_eval_classifier/best_classifier.pt

OUTPUT_DIR=./outputs/aireadi_eval_classifier

python eval_classifier_generated_vs_real.py \
    --data_root ${DATA_ROOT} \
    --participants_tsv_path ${PARTICIPANTS_TSV} \
    --real_split test \
    --gen_ckpt_path ${GEN_CKPT} \
    --gen_model DriftDiT-Tiny \
    --use_ema \
    --cfg_alpha 1.0 \
    --clf_ckpt_path ${CLF_CKPT} \
    --clf_hidden_channels 32 \
    --clf_dropout 0.1 \
    --num_classes 4 \
    --samples_per_class 1000 \
    --ts_seq_len 288 \
    --ts_delay 18 \
    --ts_embedding 18 \
    --img_size 18 \
    --in_channels 1 \
    --window_mode daily \
    --window_stride 128 \
    --daily_min_events 288 \
    --batch_size 256 \
    --gen_batch_size 256 \
    --num_workers 4 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --output_json generated_vs_real_classifier_gap.json
