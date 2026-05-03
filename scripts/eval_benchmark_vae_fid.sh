#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/eval_benchmark_vae_fid.py" \
  --num_samples 2000 \
  --seed 2026 \
  --device cuda \
  --vae_ckpt_root /playpen-shared/haochenz/ImagenFew/fid_vae_ckpts \
  "$@"
