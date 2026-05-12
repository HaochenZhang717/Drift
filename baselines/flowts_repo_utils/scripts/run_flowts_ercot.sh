#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/Drift/baselines/flowts_repo_utils"
cd "${ROOT_DIR}"

export PYTHONPATH="/playpen-shared/haochenz/Drift:${PYTHONPATH:-}"
export hucfg_num_steps="${hucfg_num_steps:-800}"
export hucfg_t_sampling="${hucfg_t_sampling:-logitnorm}"
export hucfg_attention_rope_use="${hucfg_attention_rope_use:-1}"

CONFIG_DIR="${ROOT_DIR}/Config/benchmark"
OUTPUT_ROOT="/mnt/unites8/playpen/haochenz/Drift/baselines/FlowTS"
mkdir -p "${OUTPUT_ROOT}"

for len in 64 128 256 512; do
  cfg="${CONFIG_DIR}/ErcotData_len${len}.yaml"
  exp_name="ErcotData_len${len}"
  export results_folder="${OUTPUT_ROOT}/${exp_name}"
  mkdir -p "${results_folder}"
  echo "Running ${exp_name} with ${cfg}"
  python main.py --name "${exp_name}" --train --config_file "${cfg}" --long_len "${len}"
done
