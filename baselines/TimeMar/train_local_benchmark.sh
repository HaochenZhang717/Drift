#!/usr/bin/env bash
set -euo pipefail

# Run TimeMar on local benchmark datasets using explicit yaml configs.
#
# Defaults run:
#   GlucoseSliding, ErcotData, HouseholdData x seq_len 64,128,256,512.
#
# Useful examples:
#   bash train_local_benchmark.sh
#   DATASETS="ErcotData" LENGTHS="64" bash train_local_benchmark.sh
#   DRY_RUN_MODEL_SIZE=1 DEVICE=cpu DATASETS="ErcotData" LENGTHS="64" bash train_local_benchmark.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATASETS="${DATASETS:-GlucoseSliding ErcotData HouseholdData}"
LENGTHS="${LENGTHS:-64 128 256 512}"
CONFIG_DIR="${CONFIG_DIR:-configs/local_benchmark}"
VQVAE_SAVE_ROOT="${VQVAE_SAVE_ROOT:-../dual_vqvae_timemar_local}"
VAR_SAVE_ROOT="${VAR_SAVE_ROOT:-../var_timemar_local}"
DEVICE="${DEVICE:-}"
DRY_RUN_MODEL_SIZE="${DRY_RUN_MODEL_SIZE:-0}"
MAX_MODEL_PARAMS="${MAX_MODEL_PARAMS:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${VQVAE_SAVE_ROOT}" "${VAR_SAVE_ROOT}"

model_size_args=()
if [[ "${DRY_RUN_MODEL_SIZE}" == "1" ]]; then
  model_size_args+=(--dry_run_model_size)
fi
if [[ -n "${MAX_MODEL_PARAMS}" ]]; then
  model_size_args+=(--max_model_params "${MAX_MODEL_PARAMS}")
fi

run_vqvae() {
  if [[ -n "${DEVICE}" ]]; then
    python train_dual_vqvae.py "$@" --device "${DEVICE}"
  else
    python train_dual_vqvae.py "$@"
  fi
}

for dataset in ${DATASETS}; do
  for length in ${LENGTHS}; do
    run_name="${dataset}_len${length}"
    vq_config="${CONFIG_DIR}/train_vq_${run_name}.yaml"
    var_config="${CONFIG_DIR}/train_var_${run_name}.yaml"
    vq_ckpt="${VQVAE_SAVE_ROOT}/vq_${run_name}/checkpoints/best.pt"
    var_save_dir="${VAR_SAVE_ROOT}/var_${run_name}"

    if [[ ! -f "${vq_config}" ]]; then
      echo "Missing VQ config: ${vq_config}" >&2
      exit 1
    fi
    if [[ ! -f "${var_config}" ]]; then
      echo "Missing VAR config: ${var_config}" >&2
      exit 1
    fi

    echo "=== [${run_name}] Stage 1: DualVQVAE ==="
    echo "Config: ${vq_config}"
    run_vqvae \
      --data "${run_name}" \
      --config "${vq_config}" \
      --save_dir "${VQVAE_SAVE_ROOT}" \
      ${model_size_args+"${model_size_args[@]}"}

    echo "=== [${run_name}] Stage 2: VAR ==="
    echo "Config: ${var_config}"
    if [[ "${DRY_RUN_MODEL_SIZE}" == "1" ]]; then
      python train_ar.py \
        --data "${dataset}" \
        --config "${var_config}" \
        --save_dir "${var_save_dir}" \
        ${model_size_args+"${model_size_args[@]}"}
    else
      python train_ar.py \
        --data "${dataset}" \
        --vqvae_path "${vq_ckpt}" \
        --config "${var_config}" \
        --save_dir "${var_save_dir}" \
        ${model_size_args+"${model_size_args[@]}"}
    fi
  done
done
