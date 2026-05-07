#!/bin/bash

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=2

DATA_ROOT=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed
PARTICIPANTS_TSV=/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv

MM_CKPT_DIR=/playpen-shared/haochenz/Drift/outputs/multimodal_jit_v1
CLF_CKPT=/playpen-shared/haochenz/Drift/outputs/aireadi_eval_classifier/best_classifier.pt
OUTPUT_DIR=/playpen-shared/haochenz/Drift/outputs/aireadi_eval_classifier/mm_ckpt_sweep

mkdir -p "${OUTPUT_DIR}"

CHECKPOINTS=(
  checkpoint_epoch100.pt
  checkpoint_epoch200.pt
  checkpoint_epoch300.pt
  checkpoint_epoch400.pt
  checkpoint_epoch500.pt
  checkpoint_epoch600.pt
  checkpoint_epoch700.pt
  checkpoint_epoch800.pt
)

EMA_WEIGHTS=(ema1 ema2)

for ckpt_name in "${CHECKPOINTS[@]}"; do
  ckpt_path="${MM_CKPT_DIR}/${ckpt_name}"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[WARN] missing checkpoint: ${ckpt_path}, skipping"
    continue
  fi

  ckpt_stem="${ckpt_name%.pt}"

  for mm_weights in "${EMA_WEIGHTS[@]}"; do
    out_json="mm_gap_${ckpt_stem}_${mm_weights}.json"

    python eval_classifier_mm_generated_vs_real.py \
      --data_root "${DATA_ROOT}" \
      --participants_tsv_path "${PARTICIPANTS_TSV}" \
      --real_split test \
      \
      --mm_ckpt_path "${ckpt_path}" \
      --mm_weights "${mm_weights}" \
      \
      --clf_ckpt_path "${CLF_CKPT}" \
      --clf_hidden_channels 32 \
      --clf_dropout 0.1 \
      \
      --num_classes 4 \
      --samples_per_class 560 \
      \
      --ts_seq_len 288 \
      --ts_delay 18 \
      --ts_embedding 18 \
      \
      --window_mode daily \
      --window_stride 288 \
      --daily_min_events 288 \
      --max_anchor_gap_minutes 10 \
      --max_window_span_hours 24 \
      --anchor_sampling_minutes 5.0 \
      --anchor_sampling_tolerance_seconds 2 \
      \
      --batch_size 256 \
      --gen_batch_size 256 \
      --num_workers 4 \
      --seed 42 \
      --num_repeats 10 \
      \
      --output_dir "${OUTPUT_DIR}" \
      --output_json "${out_json}"
  done
done
