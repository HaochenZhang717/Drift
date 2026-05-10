#!/usr/bin/env bash
#SBATCH --job-name=drift
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

ROOT_DIR="/playpen-shared/haochenz/Drift"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# =========================
# Core dataset/runtime config (override via env vars in submit script)
# =========================
DATASETS_DIR=${DATASETS_DIR:-/mnt/unites8/playpen/haochenz/Time_Series_Datasets}
DATA_BACKEND=${DATA_BACKEND:-ErcotData}
DATASET_NAME=${DATASET_NAME:-ErcotData}
REL_PATH=${REL_PATH:-ERCOT_merged.csv}
REL_PATH_TRAIN=${REL_PATH_TRAIN:-}
REL_PATH_VALID=${REL_PATH_VALID:-}
IN_CHANNEL=${IN_CHANNEL:-8}
ONE_CHANNEL=${ONE_CHANNEL:-0}

# =========================
# Training config
# =========================
BATCH_SIZE=${BATCH_SIZE:-512}
BS_POS=${BS_POS:-1024}
EPOCHS=${EPOCHS:-1000}
TS_LEN=${TS_LEN:-64}
IMG_SIZE=${IMG_SIZE:-8}
STRIDE=${STRIDE:-1}
DRIFT_LOSS_MODE=${DRIFT_LOSS_MODE:-time_series}

PROJECT_ROOT="/playpen-shared/haochenz/Drift"
OUTPUT_DIR="/mnt/unites8/playpen/haochenz/Drift/drift_outputs/benchmark${TS_LEN}/${DATASET_NAME}"
DATASET_LOWER=$(echo "${DATASET_NAME}" | tr '[:upper:]' '[:lower:]')
VAE_ROOT=${VAE_ROOT:-/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_${DATASET_LOWER}_${TS_LEN}}
VAE_CKPT_NAME=${VAE_CKPT_NAME:-best.pt}

case "${DATASET_NAME}" in
  ErcotData) DEFAULT_VQVAE_DATASET_SLUG=ercot ;;
  HouseholdData) DEFAULT_VQVAE_DATASET_SLUG=household ;;
  GlucoseSliding) DEFAULT_VQVAE_DATASET_SLUG=glucosesliding ;;
  *) DEFAULT_VQVAE_DATASET_SLUG="${DATASET_LOWER}" ;;
esac
VQVAE_DATASET_SLUG=${VQVAE_DATASET_SLUG:-${DEFAULT_VQVAE_DATASET_SLUG}}
VQVAE_ROOT=${VQVAE_ROOT:-/mnt/unites8/playpen/haochenz/Drift/vqvae_ckpts/benchmark_${VQVAE_DATASET_SLUG}_${TS_LEN}/${DATASET_NAME}}
VQVAE_CKPT_NAME=${VQVAE_CKPT_NAME:-best.pt}
VQVAE_CKPT_PATH=${VQVAE_CKPT_PATH:-${VQVAE_ROOT}/${VQVAE_CKPT_NAME}}
VQVAE_HIDDEN_SIZE=${VQVAE_HIDDEN_SIZE:-32}
VQVAE_NUM_LAYERS=${VQVAE_NUM_LAYERS:-1}
VQVAE_CODE_DIM=${VQVAE_CODE_DIM:-8}
VQVAE_NUM_CODES=${VQVAE_NUM_CODES:-150}
VQVAE_LATENT_DOWNSAMPLE=${VQVAE_LATENT_DOWNSAMPLE:-16}
VQVAE_DECODER_UPSAMPLE_RATE=${VQVAE_DECODER_UPSAMPLE_RATE:-4}
VQVAE_DROPOUT=${VQVAE_DROPOUT:-0.1}
VQVAE_COMMITMENT_WEIGHT=${VQVAE_COMMITMENT_WEIGHT:-0.25}

declare -a EXTRA_ARGS=()
if [[ "${ONE_CHANNEL}" == "1" ]]; then
  EXTRA_ARGS+=(--one_channel)
  IN_CHANNEL=1
fi

if [[ -n "${REL_PATH}" ]]; then
  EXTRA_ARGS+=(--rel_path "${REL_PATH}")
fi
if [[ -n "${REL_PATH_TRAIN}" ]]; then
  EXTRA_ARGS+=(--rel_path_train "${REL_PATH_TRAIN}")
fi
if [[ -n "${REL_PATH_VALID}" ]]; then
  EXTRA_ARGS+=(--rel_path_valid "${REL_PATH_VALID}")
fi

python benchmarking_drift.py \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --num_workers 16 \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --eval_splits test \
  --model DriftDiT-Tiny \
  --img_size "${IMG_SIZE}" \
  --in_channels "${IN_CHANNEL}" \
  --batch_n_pos "${BS_POS}" \
  --batch_n_neg "${BATCH_SIZE}" \
  --temperatures 0.02,0.05,0.2 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --ema_decay 0.999 \
  --warmup_steps 1000 \
  --drift_loss_mode "${DRIFT_LOSS_MODE}" \
  --queue_size 1280 \
  --vqvae_ckpt_path "${VQVAE_CKPT_PATH}" \
  --vqvae_hidden_size "${VQVAE_HIDDEN_SIZE}" \
  --vqvae_num_layers "${VQVAE_NUM_LAYERS}" \
  --vqvae_code_dim "${VQVAE_CODE_DIM}" \
  --vqvae_num_codes "${VQVAE_NUM_CODES}" \
  --vqvae_latent_downsample "${VQVAE_LATENT_DOWNSAMPLE}" \
  --vqvae_decoder_upsample_rate "${VQVAE_DECODER_UPSAMPLE_RATE}" \
  --vqvae_dropout "${VQVAE_DROPOUT}" \
  --vqvae_commitment_weight "${VQVAE_COMMITMENT_WEIGHT}" \
  --ts_seq_len "${TS_LEN}" \
  --ts_delay "${IMG_SIZE}" \
  --ts_embedding "${IMG_SIZE}" \
  --window_stride "${STRIDE}" \
  --ts_stride "${STRIDE}" \
  --stride "${STRIDE}" \
  --dataset_name "${DATASET_NAME}" \
  --data "${DATA_BACKEND}" \
  --datasets_dir "${DATASETS_DIR}" \
  --eval_metrics vaeFID \
  --eval_num_samples 2000 \
  --eval_step_interval 500 \
  --vae_ckpt_root "${VAE_ROOT}" \
  --vae_ckpt_name "${VAE_CKPT_NAME}" \
  --wandb \
  --wandb_project BenchmarkingDrift \
  --wandb_run_name "${DATASET_NAME}_${TS_LEN}_S${STRIDE}_C${IN_CHANNEL}" \
  "${EXTRA_ARGS[@]}"
