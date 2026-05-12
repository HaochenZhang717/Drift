#!/usr/bin/env bash
set -euo pipefail

# Run the 12 SDFlow benchmark settings:
#   datasets: GlucoseSliding, ErcotData, HouseholdData
#   lengths : 64, 128, 256, 512
#
# Usage examples:
#   bash scripts/run_benchmark_12.sh
#   PHASE=stage1 bash scripts/run_benchmark_12.sh
#   PHASE=stage2 DATASETS=glucosesliding LENGTHS=128 bash scripts/run_benchmark_12.sh
#   DRY_RUN=1 bash scripts/run_benchmark_12.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SDFLOW_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PHASE="${PHASE:-all}"              # all | stage1 | stage2
DATASETS="${DATASETS:-glucosesliding ercot household}"
LENGTHS="${LENGTHS:-64 128 256 512}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"

GLUCOSE_DATA_DIR="${GLUCOSE_DATA_DIR:-../../AI-READI}"
LOCAL_TS_DATA_DIR="${LOCAL_TS_DATA_DIR:-/mnt/unites8/playpen/haochenz/Time_Series_Datasets}"

VQ_BATCH_SIZE="${VQ_BATCH_SIZE:-128}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"

VQ_TOTAL_ITER="${VQ_TOTAL_ITER:-100000}"
VQ_WARM_UP_ITER="${VQ_WARM_UP_ITER:-1000}"
VQ_LR="${VQ_LR:-1e-4}"
VQ_LR_SCHEDULER="${VQ_LR_SCHEDULER:-200000}"
VQ_EVAL_ITER="${VQ_EVAL_ITER:-5000}"
VQ_PRINT_ITER="${VQ_PRINT_ITER:-200}"
VQ_WIDTH="${VQ_WIDTH:-512}"
VQ_CODE_DIM="${VQ_CODE_DIM:-512}"
VQ_NB_CODE="${VQ_NB_CODE:-512}"
VQ_DOWN_T="${VQ_DOWN_T:-2}"
VQ_DEPTH="${VQ_DEPTH:-3}"
VQ_DILATION_GROWTH_RATE="${VQ_DILATION_GROWTH_RATE:-3}"
VQ_COMMIT="${VQ_COMMIT:-0.001}"
QUANTIZER="${QUANTIZER:-ema_reset_sim}"

FLOW_D_MODEL="${FLOW_D_MODEL:-1024}"
FLOW_N_LAYERS="${FLOW_N_LAYERS:-3}"
FLOW_NUM_HEADS="${FLOW_NUM_HEADS:-16}"
FLOW_DROPOUT="${FLOW_DROPOUT:-0.1}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.01}"
FLOW_LAMBDA_MEAN="${FLOW_LAMBDA_MEAN:-0.1}"
FLOW_LAMBDA_STD="${FLOW_LAMBDA_STD:-10.0}"
FLOW_LR="${FLOW_LR:-1e-4}"
FLOW_LR_UV="${FLOW_LR_UV:-1e-3}"
FLOW_PRINT_INTERVAL="${FLOW_PRINT_INTERVAL:-200}"
FLOW_EVAL_INTERVAL="${FLOW_EVAL_INTERVAL:-20000}"
FLOW_EVAL_STEPS="${FLOW_EVAL_STEPS:-20}"
FLOW_MAX_ITERS="${FLOW_MAX_ITERS:-}"
FLOW_NUM_SAMPLES="${FLOW_NUM_SAMPLES:-50000}"
FLOW_SAMPLE_BATCH_SIZE="${FLOW_SAMPLE_BATCH_SIZE:-128}"

run_cmd() {
  printf '\n'
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

dataset_args() {
  local dataset="$1"
  case "${dataset}" in
    glucosesliding|glucose)
      echo "--dataname glucose --datasets-dir ${GLUCOSE_DATA_DIR} --rel-path-train glucose_train.parquet --rel-path-valid glucose_valid.parquet --stride 128"
      ;;
    ercot)
      echo "--dataname ercot --datasets-dir ${LOCAL_TS_DATA_DIR} --rel-path ERCOT_merged.csv --stride 1"
      ;;
    household)
      echo "--dataname household --datasets-dir ${LOCAL_TS_DATA_DIR} --rel-path HouseHold_6.csv --stride 10 --window-stride 10 --ts-stride 10"
      ;;
    *)
      echo "Unknown dataset '${dataset}'. Expected: glucosesliding, ercot, household." >&2
      exit 2
      ;;
  esac
}

dataset_label() {
  local dataset="$1"
  case "${dataset}" in
    glucosesliding|glucose) echo "glucosesliding" ;;
    ercot) echo "ercot" ;;
    household) echo "household" ;;
  esac
}

flow_rank_for_length() {
  local length="$1"
  case "${length}" in
    64) echo 256 ;;
    128) echo 512 ;;
    256|512) echo 1024 ;;
    *)
      echo "Unsupported length '${length}'. Expected: 64, 128, 256, 512." >&2
      exit 2
      ;;
  esac
}

flow_iters_for_length() {
  local length="$1"
  if [[ -n "${FLOW_MAX_ITERS}" ]]; then
    echo "${FLOW_MAX_ITERS}"
    return
  fi
  case "${length}" in
    64) echo 120000 ;;
    128|256|512) echo 100000 ;;
    *)
      echo "Unsupported length '${length}'. Expected: 64, 128, 256, 512." >&2
      exit 2
      ;;
  esac
}

run_stage1() {
  local dataset="$1"
  local length="$2"
  local label
  label="$(dataset_label "${dataset}")"

  # shellcheck disable=SC2206
  local data_args=( $(dataset_args "${dataset}") )

  run_cmd python stage1_vq/train_vq.py \
    "${data_args[@]}" \
    --window-size "${length}" \
    --batch-size "${VQ_BATCH_SIZE}" \
    --width "${VQ_WIDTH}" \
    --lr "${VQ_LR}" \
    --total-iter "${VQ_TOTAL_ITER}" \
    --warm-up-iter "${VQ_WARM_UP_ITER}" \
    --lr-scheduler "${VQ_LR_SCHEDULER}" \
    --code-dim "${VQ_CODE_DIM}" \
    --nb-code "${VQ_NB_CODE}" \
    --down-t "${VQ_DOWN_T}" \
    --depth "${VQ_DEPTH}" \
    --dilation-growth-rate "${VQ_DILATION_GROWTH_RATE}" \
    --out-dir "./output/output_${label}" \
    --vq-act relu \
    --quantizer "${QUANTIZER}" \
    --exp-name "VQVAE${length}" \
    --commit "${VQ_COMMIT}" \
    --print-iter "${VQ_PRINT_ITER}" \
    --eval-iter "${VQ_EVAL_ITER}" \
    --gpu "${GPU}" \
    --device "${DEVICE}" \
    --skip-eval
}

run_stage2() {
  local dataset="$1"
  local length="$2"
  local label rank max_iters
  label="$(dataset_label "${dataset}")"
  rank="$(flow_rank_for_length "${length}")"
  max_iters="$(flow_iters_for_length "${length}")"

  # shellcheck disable=SC2206
  local data_args=( $(dataset_args "${dataset}") )

  run_cmd python stage2_flow/train_sdflow.py \
    --vqvae_ckpt "./output/output_${label}/VQVAE${length}/net_best_ds.pth" \
    "${data_args[@]}" \
    --output_dir "./checkpoints_${label}_${length}" \
    --window_size "${length}" \
    --down_t "${VQ_DOWN_T}" \
    --quantizer "${QUANTIZER}" \
    --rank "${rank}" \
    --d_model "${FLOW_D_MODEL}" \
    --n_layers "${FLOW_N_LAYERS}" \
    --num_heads "${FLOW_NUM_HEADS}" \
    --dropout "${FLOW_DROPOUT}" \
    --noise_std "${FLOW_NOISE_STD}" \
    --lambda_mean "${FLOW_LAMBDA_MEAN}" \
    --lambda_std "${FLOW_LAMBDA_STD}" \
    --batch_size "${FLOW_BATCH_SIZE}" \
    --lr "${FLOW_LR}" \
    --lr_uv "${FLOW_LR_UV}" \
    --max_iters "${max_iters}" \
    --print_interval "${FLOW_PRINT_INTERVAL}" \
    --eval_interval "${FLOW_EVAL_INTERVAL}" \
    --eval_steps "${FLOW_EVAL_STEPS}" \
    --num_samples "${FLOW_NUM_SAMPLES}" \
    --sample_batch_size "${FLOW_SAMPLE_BATCH_SIZE}" \
    --device "${DEVICE}"
}

for dataset in ${DATASETS}; do
  for length in ${LENGTHS}; do
    echo "========================================"
    echo "SDFlow ${dataset} length=${length} phase=${PHASE}"
    echo "========================================"

    case "${PHASE}" in
      all)
        run_stage1 "${dataset}" "${length}"
        run_stage2 "${dataset}" "${length}"
        ;;
      stage1)
        run_stage1 "${dataset}" "${length}"
        ;;
      stage2)
        run_stage2 "${dataset}" "${length}"
        ;;
      *)
        echo "Unknown PHASE='${PHASE}'. Expected: all, stage1, stage2." >&2
        exit 2
        ;;
    esac
  done
done
