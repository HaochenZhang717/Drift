#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
IMG_SIZE=12
TS_LEN=128
DATA_DIR="/mnt/unites8/playpen/haochenz/Time_Series_Datasets"
#DATA_DIR="/Users/zhc/Documents/Time_Series_Datasets"

python train_mm_jepa_jit.py \
    --dataset_name "HouseholdData" \
    --data "HouseholdData" \
    --datasets_dir ${DATA_DIR} \
    --rel_path "HouseHold_6.csv" \
    \
    --output_dir "/mnt/unites8/playpen/haochenz/Drift/debug_outputs/mm_jepa_jit_exp1" \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 1 \
    --pin_memory \
    --seed 42 \
    --device "auto" \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --save_interval 10 \
    \
    --input_channels 6 \
    --num_modalities 6 \
    --modality_channel_splits "1,1,1,1,1,1" \
    \
    --ts_seq_len ${TS_LEN} \
    --window_stride 10 \
    --ts_stride 10 \
    --stride 10 \
    --ts_delay ${IMG_SIZE} \
    --ts_embedding ${IMG_SIZE} \
    \
    --jepa_hidden_size 32 \
    --jepa_encoder_layers 2 \
    --jepa_embed_dim 4 \
    --jepa_latent_downsample 16 \
    --jepa_encoder_dropout 0.0 \
    --jepa_predictor_dim 64 \
    --jepa_predictor_layers 3 \
    --jepa_predictor_heads 4 \
    --jepa_predictor_mlp_ratio 4.0 \
    --jepa_predictor_dropout 0.0 \
    --jepa_ema_momentum 0.996 \
    --jepa_max_len ${TS_LEN} \
    \
    --img_size ${IMG_SIZE} \
    --patch_size 2 \
    --hidden_channels 128 \
    --depth 3 \
    --num_heads 4 \
    --attn_dropout 0.0 \
    --proj_dropout 0.0 \
    --bottleneck_dim 64 \
    --in_context_start 0 \
    --P_mean -1.2 \
    --P_std 1.2 \
    --t_eps 1e-5 \
    --noise_scale 1.0 \
    --ema_decay1 0.999 \
    --ema_decay2 0.9999 \
    --sampling_method "heun" \
    --num_sampling_steps 50 \
    --cfg 1.0 \
    --interval_min 0.0 \
    --interval_max 1.0 \
    \
    --imputation_eval_interval 100 \
    \
    --wandb_project "mm-jepa-jit" \
    --wandb_run_name "debug" \
    --wandb_entity "" \
    --wandb_mode "online"