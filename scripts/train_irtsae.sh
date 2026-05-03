python train_irregular_ts_ae.py \
  --data_root /Users/zhc/Downloads/AI-READI-processed \
  --participants_tsv_path /Users/zhc/Downloads/AI-READI/participants.tsv \
  --train_modality "heart_rate" \
  --use_aligned_modality \
  --train_split train \
  --val_split valid \
  \
  --anchor_modality "glucose" \
  --ts_seq_len 288 \
  --daily_min_events 288 \
  --max_anchor_gap_minutes 10 \
  --max_window_span_hours 24 \
  --anchor_sampling_minutes 5.0 \
  --anchor_sampling_tolerance_seconds 2 \
  --max_missing_ratio 0.5 \
  \
  --d_model 128 \
  --n_head 32 \
  --num_layers 3 \
  \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --input_random_drop_prob 0.5 \
  \
  --num_workers 1 \
  -seed 0 \
  \
  --save_dir "./outputs/irregular_ts_ae/heart_rate" \
  --save_name "best.pt" \
  \
  --wandb_project irregular-ts-ae

