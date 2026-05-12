#!/usr/bin/env bash
set -euo pipefail

# 这个文件是我用来evaluate ETTh2, ETTm1, 和 ETTm2的结果的。
# Evaluate discriminative score for benchmark256 checkpoints.
# Default scan paths:
#   <ROOT_DIR>/ETTh2/ETTh2
#   <ROOT_DIR>/ETTm1/ETTm1
#   <ROOT_DIR>/ETTm2/ETTm2

ROOT_DIR="${1:-/mnt/unites8/playpen/haochenz/Drift/drift_outputs/benchmark256}"
NUM_RUNS="${NUM_RUNS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
USE_EMA="${USE_EMA:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_drift_discriminative_ckpts.py}"
OUT_NAME="${OUT_NAME:-discriminative_ckpt_results.jsonl}"

TARGET_DATASETS=("ETTh2" "ETTm1" "ETTm2")

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] Cannot find evaluation script: $EVAL_SCRIPT"
  echo "        Please run this shell under the repo root or set EVAL_SCRIPT=/abs/path/to/eval_drift_discriminative_ckpts.py"
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[ERROR] ROOT_DIR does not exist: $ROOT_DIR"
  exit 1
fi

CKPT_DIRS=()
for ds in "${TARGET_DATASETS[@]}"; do
  ckpt_dir="$ROOT_DIR/$ds/$ds"
  if [[ -d "$ckpt_dir" ]]; then
    CKPT_DIRS+=("$ckpt_dir")
  else
    echo "[WARN] Missing dataset checkpoint dir: $ckpt_dir"
  fi
done

if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "[WARN] No valid checkpoint directories found under: $ROOT_DIR"
  exit 0
fi

echo "[INFO] Found ${#CKPT_DIRS[@]} candidate directories"
echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] NUM_RUNS=$NUM_RUNS, BATCH_SIZE=$BATCH_SIZE, NUM_WORKERS=$NUM_WORKERS, USE_EMA=$USE_EMA"

for ckpt_dir in "${CKPT_DIRS[@]}"; do
  shopt -s nullglob
  ckpts=("$ckpt_dir"/*.pt)
  shopt -u nullglob

  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "[SKIP] $ckpt_dir (no *.pt found)"
    continue
  fi

  echo "[RUN ] Evaluating: $ckpt_dir"

  cmd=(
    "$PYTHON_BIN" "$EVAL_SCRIPT"
    --checkpoint_dir "$ckpt_dir"
    --num_runs "$NUM_RUNS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --output_jsonl "$OUT_NAME"
  )

  if [[ "$USE_EMA" == "1" ]]; then
    cmd+=(--use_ema)
  fi

  "${cmd[@]}"
  echo "[DONE] $ckpt_dir/$OUT_NAME"
  echo
done

echo "[ALL DONE]"
