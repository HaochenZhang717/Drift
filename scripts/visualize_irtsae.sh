python visualize_irtsae_without_input_mask_imputed.py \
  --ckpt_path /playpen-shared/haochenz/Drift/outputs/irregular_ts_ae/calorie/best.pt \
  --num_samples 4 \
  --split test

python visualize_irtsae_without_input_mask_imputed.py \
  --ckpt_path /playpen-shared/haochenz/Drift/outputs/irregular_ts_ae/heart_rate/best.pt \
  --num_samples 4 \
  --split test

python visualize_irtsae_without_input_mask_imputed.py \
  --ckpt_path /playpen-shared/haochenz/Drift/outputs/irregular_ts_ae/physical_activity/best.pt \
  --num_samples 4 \
  --split test

python visualize_irtsae_without_input_mask_imputed.py \
  --ckpt_path /playpen-shared/haochenz/Drift/outputs/irregular_ts_ae/respiratory_rate/best.pt \
  --num_samples 4 \
  --split test

