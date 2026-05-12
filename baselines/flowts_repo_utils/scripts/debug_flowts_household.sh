#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/zhc/Documents/PhD/projects/drifting-model/baselines/flowts_repo_utils"
cd "${ROOT_DIR}"

export PYTHONPATH="/Users/zhc/Documents/PhD/projects/drifting-model:${PYTHONPATH:-}"
export hucfg_num_steps="${hucfg_num_steps:-800}"
export hucfg_t_sampling="${hucfg_t_sampling:-logitnorm}"
export hucfg_attention_rope_use="${hucfg_attention_rope_use:-1}"

CONFIG_DIR="${ROOT_DIR}/Config/benchmark"
OUTPUT_ROOT="/Users/zhc/Documents/PhD/projects/debug_flowts_outputs"
mkdir -p "${OUTPUT_ROOT}"

for len in 64; do
  cfg="${CONFIG_DIR}/HouseholdData_len${len}.yaml"
  exp_name="HouseholdData_len${len}"
  export results_folder="${OUTPUT_ROOT}/${exp_name}"
  mkdir -p "${results_folder}"
  echo "Running ${exp_name} with ${cfg}"
  python main.py --name "${exp_name}" --train --config_file "${cfg}" --long_len "${len}"
done
