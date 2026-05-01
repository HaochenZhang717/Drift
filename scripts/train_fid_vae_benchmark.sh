

python train_fid_vae_benchmark.py \
  --dataset_name "${DATASET_NAME}" \
  --data "ETTm1" \
  --datasets_dir "ETTm1" \
  --rel_path "TSF/ETT-small/ETTm1.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 200 \
  --save_dir ./fid_vae_ckpts/benchmark
