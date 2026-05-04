#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/Drift"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/baselines/ImagenFew/configs/ImagenTime/glucose.yaml}"
SUBSET_P="${SUBSET_P:-1.0}"

python "${ROOT_DIR}/baselines/ImagenFew/run.py" \
  --subset_p "${SUBSET_P}" \
  --config "${CONFIG_FILE}"
