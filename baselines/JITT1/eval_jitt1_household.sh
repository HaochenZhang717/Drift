#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/mnt/unites8/playpen/haochenz/JITT1_outputs"

for SEQ_LEN in 64 128 256 512; do
  RUN_DIR="${BASE_DIR}/HouseholdData_len${SEQ_LEN}"
  python /playpen-shared/haochenz/Drift/eval_jitt1_discriminative_ckpts.py \
    --run_dir "${RUN_DIR}" \
    --output_jsonl "${RUN_DIR}/discriminative_scores_ema.jsonl" \
    --num_runs 10 \
    --gen_batch_size 256 \
    --real_batch_size 256 \
    --num_workers 8 \
    --device cuda

done
