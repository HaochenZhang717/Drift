python /Users/zhc/Documents/PhD/projects/drifting-model/analyze_aireadi_daily_irregularity.py \
  --root /Users/zhc/Downloads/AI-READI-processed \
  --participants_tsv_path /Users/zhc/Downloads/AI-READI/participants.tsv \
  --split train \
  --window_size 288 \
  --daily_min_events 288 \
  --csv_out /Users/zhc/Documents/PhD/projects/drifting-model/outputs/aireadi_daily_irregularity.csv \
  --test_calorie_alignment \
  --test_samples 300 \
  --raw_values \
  --test_shuffle

