#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Users/zhc/Documents/PhD/projects/drifting-model}
DATASETS_DIR=${DATASETS_DIR:-/Users/zhc/Documents/Time_Series_Datasets}
REL_PATH=${REL_PATH:-HouseHold_6.csv}
TS_SEQ_LEN=${TS_SEQ_LEN:-256}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-100}
SAVE_DIR=${SAVE_DIR:-${PROJECT_ROOT}/fid_vae_ckpts/local_timeseries_${TS_SEQ_LEN}}
DEVICE=${DEVICE:-cuda}

python "${PROJECT_ROOT}/train_fid_vae_benchmark.py" \
  --dataset_name "HouseholdData" \
  --data "HouseholdData" \
  --datasets_dir "${DATASETS_DIR}" \
  --rel_path "${REL_PATH}" \
  --ts_seq_len "${TS_SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --save_dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  "$@"
