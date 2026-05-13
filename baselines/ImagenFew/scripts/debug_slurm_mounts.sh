#!/usr/bin/env bash
#SBATCH --job-name=debug-mount
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

# Intentionally avoid set -e, so we can keep collecting diagnostics after failures.
set -u

TARGET_DIR=${1:-/mnt/unites8/playpen/haochenz/Drift/baselines/FlowTS}
ROOT_DIR="/playpen-shared/haochenz/Drift"
CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"

echo "===== BASIC INFO ====="
date
hostname
whoami
id
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-N/A}"
echo "TARGET_DIR=${TARGET_DIR}"
echo

echo "===== DIRECTORY CHECKS ====="
ls -ld /mnt || true
ls -ld /mnt/unites8 || true
ls -ld /mnt/unites8/playpen || true
ls -ld /mnt/unites8/playpen/haochenz || true
ls -ld /playpen-shared || true
ls -ld /playpen-shared/haochenz || true
echo

echo "===== MOUNT INFO ====="
mount | grep -E "(/mnt|/playpen-shared|unites8|nfs)" || true
df -h /mnt 2>/dev/null || true
df -h /mnt/unites8 2>/dev/null || true
df -h /playpen-shared 2>/dev/null || true
echo

echo "===== WRITE TESTS ====="
echo "[test] mkdir -p ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}" 2>&1
echo "[exit] $?"

echo "[test] touch ${TARGET_DIR}/__debug_write_test__${SLURM_JOB_ID:-manual}.txt"
touch "${TARGET_DIR}/__debug_write_test__${SLURM_JOB_ID:-manual}.txt" 2>&1
echo "[exit] $?"
ls -l "${TARGET_DIR}" 2>/dev/null | tail -n 5 || true
echo

echo "===== CONDA / LMOD CHECK ====="
echo "CONDA_BIN=${CONDA_BIN}"
ls -l "${CONDA_BIN}" || true
ls -l /playpen-shared/software/lmod/lmod/lmod/libexec/addto || true
echo "[test] conda shell hook"
eval "$("${CONDA_BIN}" shell.bash hook)" 2>&1
echo "[exit] $?"
echo "[test] conda activate vlm"
conda activate vlm 2>&1
echo "[exit] $?"
which python || true
python -V || true
echo

echo "===== DONE ====="
