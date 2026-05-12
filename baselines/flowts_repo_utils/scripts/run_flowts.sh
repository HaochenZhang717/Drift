#!/usr/bin/env bash
#SBATCH --job-name=flowts
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

ROOT_DIR="/playpen-shared/haochenz/Drift/baselines/flowts_repo_utils"
cd "${ROOT_DIR}"

CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate vlm

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export PYTHONPATH="/playpen-shared/haochenz/Drift:${PYTHONPATH:-}"
export hucfg_num_steps="${hucfg_num_steps:-800}"
export hucfg_t_sampling="${hucfg_t_sampling:-logitnorm}"

CONFIG_FILE=${1:-/playpen-shared/haochenz/Drift/baselines/flowts_repo_utils/Config/benchmark/ErcotData_len64.yaml}
EXP_NAME=$(basename "${CONFIG_FILE}" .yaml)
export results_folder="/mnt/unites8/playpen/haochenz/Drift/baselines/FlowTS/${EXP_NAME}"
mkdir -p "${results_folder}"

python main.py \
  --name "${EXP_NAME}" \
  --train \
  --config_file "${CONFIG_FILE}" \
  --long_len "$(echo "${EXP_NAME}" | awk -F'len' '{print $2}')"
