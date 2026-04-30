#!/usr/bin/env bash
#SBATCH --job-name=drift
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  else
    echo "Could not find a usable conda binary." >&2
    exit 1
  fi
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# =========================
# 基本设置
# =========================
#export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed
PARTICIPANTS_TSV=/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv


PROJECT_ROOT=/playpen-shared/haochenz/Drift
OUTPUT_DIR=${PROJECT_ROOT}/outputs/glucose_cls_cond_cfg
VAE_ROOT=${PROJECT_ROOT}/fid_vae_ckpts


# =========================
# 训练参数
# =========================
EPOCHS=500

# =========================
# 运行
# =========================
python train_ts_cond_daily.py \
    --output_dir ${OUTPUT_DIR} \
    --data_root ${DATA_ROOT} \
    --num_workers 16 \
    --batch_size 256 \
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
    --label_dropout 0.1 \
    --alpha_min 1.0 \
    --alpha_max 3.0 \
    --cfg_sample_alpha 1.5 \
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
    --eval_per_class_samples 1000 \
    --eval_step_interval 500 \
    \
    --vae_ckpt_root ${VAE_ROOT} \
    \
    --wandb \
    --wandb_project drifting-model-ts \
    --wandb_run_name glucose_cls_cond_cfg
