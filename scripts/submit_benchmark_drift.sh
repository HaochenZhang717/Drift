#!/usr/bin/env bash
set -euo pipefail

# Submit Drift benchmarks on GlucoseSliding / ErcotData / HouseholdData.
# Edit TS_LENGTHS and (optionally) DRIFT_LOSS_MODE below as needed.

TS_LENGTHS=(64 128 256 512)
DRIFT_LOSS_MODE=${DRIFT_LOSS_MODE:-time_series}


for TSLEN in "${TS_LENGTHS[@]}"; do
  IMG_SIZE=8
  if [[ "${TSLEN}" == "128" ]]; then IMG_SIZE=12; fi
  if [[ "${TSLEN}" == "256" ]]; then IMG_SIZE=16; fi
  if [[ "${TSLEN}" == "512" ]]; then IMG_SIZE=24; fi

  # GlucoseSliding (explicit train/valid parquet)
  DATASETS_DIR=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed \
  DATA_BACKEND=GlucoseSliding \
  DATASET_NAME=GlucoseSliding \
  REL_PATH= \
  REL_PATH_TRAIN=glucose_train.parquet \
  REL_PATH_VALID=glucose_valid.parquet \
  STRIDE=128 \
  TS_LEN="${TSLEN}" \
  IMG_SIZE="${IMG_SIZE}" \
  IN_CHANNEL=1 \
  DRIFT_LOSS_MODE="${DRIFT_LOSS_MODE}" \
  VAE_ROOT=/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_glucosesliding_${TSLEN} \
  VAE_CKPT_NAME=best.pt \
  sbatch scripts/benchmark_drift.sh

  # ERCOT
  DATASETS_DIR=/mnt/unites8/playpen/haochenz/Time_Series_Datasets \
  DATA_BACKEND=ErcotData \
  DATASET_NAME=ErcotData \
  REL_PATH=ERCOT_merged.csv \
  REL_PATH_TRAIN= \
  REL_PATH_VALID= \
  STRIDE=1 \
  TS_LEN="${TSLEN}" \
  IMG_SIZE="${IMG_SIZE}" \
  IN_CHANNEL=1 \
  DRIFT_LOSS_MODE="${DRIFT_LOSS_MODE}" \
  VAE_ROOT=/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_ercot_${TSLEN} \
  VAE_CKPT_NAME=best.pt \
  sbatch scripts/benchmark_drift.sh

  # Household
  DATASETS_DIR=/mnt/unites8/playpen/haochenz/Time_Series_Datasets \
  DATA_BACKEND=HouseholdData \
  DATASET_NAME=HouseholdData \
  REL_PATH=HouseHold_6.csv \
  REL_PATH_TRAIN= \
  REL_PATH_VALID= \
  STRIDE=10 \
  TS_LEN="${TSLEN}" \
  IMG_SIZE="${IMG_SIZE}" \
  IN_CHANNEL=6 \
  DRIFT_LOSS_MODE="${DRIFT_LOSS_MODE}" \
  VAE_ROOT=/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_household_${TSLEN} \
  VAE_CKPT_NAME=best.pt \
  sbatch scripts/benchmark_drift.sh
done


##!/usr/bin/env bash
#set -euo pipefail
#
## Submit Drift benchmarks on GlucoseSliding / ErcotData / HouseholdData.
## Edit TS_LENGTHS and (optionally) DRIFT_LOSS_MODE below as needed.
#
##TS_LENGTHS=(64 128 256 512)
#TS_LENGTHS=(64)
#DRIFT_LOSS_MODE=${DRIFT_LOSS_MODE:-time_series}
#
#for TSLEN in "${TS_LENGTHS[@]}"; do
#  IMG_SIZE=8
#  if [[ "${TSLEN}" == "128" ]]; then IMG_SIZE=12; fi
#  if [[ "${TSLEN}" == "256" ]]; then IMG_SIZE=16; fi
#  if [[ "${TSLEN}" == "512" ]]; then IMG_SIZE=24; fi
#
#  # GlucoseSliding (explicit train/valid parquet)
#  DATASETS_DIR=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed \
#  DATA_BACKEND=GlucoseSliding \
#  DATASET_NAME=GlucoseSliding \
#  REL_PATH= \
#  REL_PATH_TRAIN=glucose_train.parquet \
#  REL_PATH_VALID=glucose_valid.parquet \
#  STRIDE=128 \
#  TS_LEN="${TSLEN}" \
#  IMG_SIZE="${IMG_SIZE}" \
#  IN_CHANNEL=1 \
#  DRIFT_LOSS_MODE="${DRIFT_LOSS_MODE}" \
#  VAE_ROOT=/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_glucosesliding_${TSLEN} \
#  VAE_CKPT_NAME=best.pt \
#  bash scripts/benchmark_drift.sh
#
#  # ERCOT
#  DATASETS_DIR=/mnt/unites8/playpen/haochenz/Time_Series_Datasets \
#  DATA_BACKEND=ErcotData \
#  DATASET_NAME=ErcotData \
#  REL_PATH=ERCOT_merged.csv \
#  REL_PATH_TRAIN= \
#  REL_PATH_VALID= \
#  STRIDE=1 \
#  TS_LEN="${TSLEN}" \
#  IMG_SIZE="${IMG_SIZE}" \
#  IN_CHANNEL=1 \
#  DRIFT_LOSS_MODE="${DRIFT_LOSS_MODE}" \
#  VAE_ROOT=/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_ercot_${TSLEN} \
#  VAE_CKPT_NAME=best.pt \
#  bash scripts/benchmark_drift.sh
#
#  # Household
#  DATASETS_DIR=/mnt/unites8/playpen/haochenz/Time_Series_Datasets \
#  DATA_BACKEND=HouseholdData \
#  DATASET_NAME=HouseholdData \
#  REL_PATH=HouseHold_6.csv \
#  REL_PATH_TRAIN= \
#  REL_PATH_VALID= \
#  STRIDE=10 \
#  TS_LEN="${TSLEN}" \
#  IMG_SIZE="${IMG_SIZE}" \
#  IN_CHANNEL=6 \
#  DRIFT_LOSS_MODE="${DRIFT_LOSS_MODE}" \
#  VAE_ROOT=/mnt/unites8/playpen/haochenz/Drift/fid_vae_ckpts/benchmark_household_${TSLEN} \
#  VAE_CKPT_NAME=best.pt \
#  bash scripts/benchmark_drift.sh
#done

