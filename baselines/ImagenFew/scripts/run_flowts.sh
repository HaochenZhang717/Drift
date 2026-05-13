#!/usr/bin/env bash
#SBATCH --job-name=flowts
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/Drift"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

OUTPUT_ROOT_PRIMARY="/playpen/haochenz/Drift/baselines/FlowTS"
OUTPUT_ROOT_FALLBACK="/mnt/unites8/playpen/haochenz/Drift/baselines/FlowTS"

HOSTNAME_SHORT="$(hostname -s)"
if [[ "${HOSTNAME_SHORT}" == "unites8" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT_PRIMARY}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT_FALLBACK}"
fi

if ! mkdir -p "${OUTPUT_ROOT}" 2>/dev/null; then
  echo "ERROR: unable to create output dir ${OUTPUT_ROOT} on host ${HOSTNAME_SHORT}"
  exit 1
fi

echo "Host=${HOSTNAME_SHORT}, using OUTPUT_ROOT=${OUTPUT_ROOT}"

CONFIG_FILE=${1:-/playpen-shared/haochenz/Drift/baselines/ImagenFew/configs/FlowTS/ErcotData_len256.yaml}

python /playpen-shared/haochenz/Drift/baselines/ImagenFew/run.py \
--subset_p 1.0 \
--log_dir "${OUTPUT_ROOT}" \
--config ${CONFIG_FILE}
