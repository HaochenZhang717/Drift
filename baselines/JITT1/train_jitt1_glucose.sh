#!/usr/bin/env bash
#SBATCH --job-name=jitt1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail


CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

for SEQ_LEN in 64 128 256 512; do
  python baselines/JITT1/train_jitt1.py \
    --datasets_dir /playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed \
    --train_on_datasets GlucoseSliding \
    --glucose_rel_path glucose_{split}.parquet \
    --ercot_rel_path ERCOT_merged.csv \
    --household_rel_path HouseHold_6.csv \
    --glucose_stride 128 \
    --ercot_stride 1 \
    --household_stride 10 \
    --seq_len ${SEQ_LEN} \
    --input_channels 1 \
    --n_heads 128 \
    --patch_size 2 \
    --patch_stride 1 \
    --n_blocks 2 2 2 2 \
    --kernel_size_large 71 71 31 31 \
    --kernel_size_small 5 \
    --ffn_ratio 1.0 \
    --downsample_ratio 1 \
    --drop_attn 0.1 \
    --drop_ffn 0.0 \
    --drop_proj 0.0 \
    --drop_head 0.0 \
    --P_mean 0.0 \
    --P_std 1.0 \
    --t_eps 1e-5 \
    --noise_scale 1.0 \
    --sampling_method heun \
    --num_sampling_steps 100 \
    --epochs 2001 \
    --batch_size 512 \
    --num_workers 16 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --val_every 1 \
    --ema_decays 0.999,0.9999 \
    --device cuda \
    --wandb_project JITT1 \
    --wandb_mode online \
    --run_name GlucoseSliding_len${SEQ_LEN} \
    --output_dir /mnt/unites8/playpen/haochenz/JITT1_outputs

done
