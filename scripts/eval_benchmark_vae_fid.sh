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
ETTM2_DIR=${ETTM2_DIR:-${PROJECT_ROOT}/logs/ImagenTime/ETTm2/1a746657-a3bf-4dc4-b145-12cf4422e977/eval_samples}
ETTH2_DIR=${ETTH2_DIR:-${PROJECT_ROOT}/logs/ImagenTime/ETTh2/075d365a-305c-4664-986a-c268eae56de6/eval_samples}
ETTM1_DIR=${ETTM1_DIR:-${PROJECT_ROOT}/logs/ImagenTime/ETTm1/e1126afd-35f2-46ee-a7e9-a0ca558476c4/eval_samples}
WEATHER_DIR=${WEATHER_DIR:-${PROJECT_ROOT}/logs/ImagenTime/Weather/80b0e9fb-777d-4282-bdbc-362cb6bd47c0/eval_samples}
AIRQUALITY_DIR=${AIRQUALITY_DIR:-${PROJECT_ROOT}/logs/ImagenTime/AirQuality/5db9c0c8-2560-4827-bd7d-259d201793ed/eval_samples}

python "${PROJECT_ROOT}/eval_benchmark_vae_fid.py" \
  --num_samples "${NUM_SAMPLES}" \
  --num_repeats "${NUM_REPEATS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --vae_ckpt_root "${VAE_ROOT}" \
  --vae_ckpt_name "${VAE_CKPT_NAME}" \
  --ettm2_dir "${ETTM2_DIR}" \
  --etth2_dir "${ETTH2_DIR}" \
  --ettm1_dir "${ETTM1_DIR}" \
  --weather_dir "${WEATHER_DIR}" \
  --airquality_dir "${AIRQUALITY_DIR}" \
  "$@"



#PROJECT_ROOT=/playpen-shared/haochenz/Drift \
#VAE_ROOT=/playpen-shared/haochenz/Drift/fid_vae_ckpts \
#VAE_CKPT_NAME=best.pt \
#python eval_benchmark_vae_fid.py \
#  --single_dataset_name glucose_daily \
#  --single_run_dir /playpen-shared/haochenz/ImagenFew/logs/ImagenTime/glucose_npy/3a02d76c-d860-44fd-974b-b3f4496cdedb \
#  --num_repeats 1 \
#  --num_samples -1
