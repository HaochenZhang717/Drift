#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/Drift/baselines/ImagenFew"
CONFIG_DIR="${ROOT_DIR}/configs/ImagenTime"

CONFIGS=(
  "${CONFIG_DIR}/ErcotData_len64.yaml"
  "${CONFIG_DIR}/ErcotData_len128.yaml"
  "${CONFIG_DIR}/ErcotData_len256.yaml"
  "${CONFIG_DIR}/ErcotData_len512.yaml"
  "${CONFIG_DIR}/HouseholdData_len64.yaml"
  "${CONFIG_DIR}/HouseholdData_len128.yaml"
  "${CONFIG_DIR}/HouseholdData_len256.yaml"
  "${CONFIG_DIR}/HouseholdData_len512.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "Submitting ${cfg}"
  CONDA_ENV=vlm sbatch "${ROOT_DIR}/scripts/run_imagentime.sh" "${cfg}"
done
