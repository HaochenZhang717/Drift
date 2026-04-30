# =========================
# 基本设置
# =========================
#export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=/Users/zhc/Downloads/AI-READI-Dataset/AI-READI-processed
PARTICIPANTS_TSV=/Users/zhc/Downloads/AI-READI/participants.tsv


PROJECT_ROOT=/playpen-shared/haochenz/Drift
OUTPUT_DIR=${PROJECT_ROOT}/outputs/glucose_uncond_daily
VAE_ROOT=${PROJECT_ROOT}/fid_vae_ckpts


# =========================
# 训练参数
# =========================
BATCH_SIZE=256
EPOCHS=500

# =========================
# 运行
# =========================
python exam_ts_uncond_daily.py \
    --output_dir ${OUTPUT_DIR} \
    --data_root ${DATA_ROOT} \
    --seed 42 \
    --num_workers 16 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    \
    --model DriftDiT-Tiny \
    --img_size 18 \
    --in_channels 1 \
    \
    --batch_n_pos 160 \
    --batch_n_neg 160 \
    --temperatures 0.02,0.05,0.2 \
    \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    \
    --ema_decay 0.999 \
    --warmup_steps 1000 \
    \
    --loss_domain time_series \
    --queue_size 1280 \
    \
    --ts_seq_len 288 \
    --ts_delay 18 \
    --ts_embedding 18 \
    \
    --window_mode daily \
    \
    --modalities glucose \
    --anchor_modality glucose \
    --target_modality glucose \
    \
    --participants_tsv_path ${PARTICIPANTS_TSV} \
    --include_participant_metadata \
    --include_study_group \
    \
    --eval_metrics vaeFID \
    --eval_num_samples 1000 \
    --eval_step_interval 500 \
    \
    --vae_ckpt_root ${VAE_ROOT} \
    \
    --wandb \
    --wandb_project drifting-model-ts \
    --wandb_run_name glucose_uncond
