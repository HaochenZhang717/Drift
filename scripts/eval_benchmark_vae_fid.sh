#!/usr/bin/env bash
set -euo pipefail

# Align with scripts/benchmark_drift.sh evaluation conventions.
PROJECT_ROOT=${PROJECT_ROOT:-/playpen-shared/haochenz/Drift}
TS_LEN=${TS_LEN:-256}
VAE_ROOT=${VAE_ROOT:-${PROJECT_ROOT}/fid_vae_ckpts/benchmark_${TS_LEN}}
VAE_CKPT_NAME=${VAE_CKPT_NAME:-last.pt}
DEVICE=${DEVICE:-cuda}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
NUM_REPEATS=${NUM_REPEATS:-10}
SEED=${SEED:-42}

python "${PROJECT_ROOT}/eval_benchmark_vae_fid.py" \
  --num_samples "${NUM_SAMPLES}" \
  --num_repeats "${NUM_REPEATS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --vae_ckpt_root "${VAE_ROOT}" \
  --vae_ckpt_name "${VAE_CKPT_NAME}" \
  "$@"



#PROJECT_ROOT=/playpen-shared/haochenz/Drift \
#VAE_ROOT=/playpen-shared/haochenz/Drift/fid_vae_ckpts \
#VAE_CKPT_NAME=best.pt \
#python eval_benchmark_vae_fid.py \
#  --single_dataset_name glucose_daily \
#  --single_run_dir /playpen-shared/haochenz/ImagenFew/logs/ImagenTime/glucose_npy/3a02d76c-d860-44fd-974b-b3f4496cdedb \
#  --num_repeats 1 \
#  --num_samples -1
