#python eval_imagentime_discriminative_samples.py \
#  --imagentime_root /mnt/unites8/playpen/haochenz/Drift/baselines/ImagenTime/ImagenTime \
#  --output_jsonl /mnt/unites8/playpen/haochenz/Drift/baselines/ImagenTime/ImagenTime/imagentime_discriminative_results.jsonl \
#  --num_runs 10 \
#  --config_root /playpen-shared/haochenz/Drift/baselines/ImagenFew/configs/ImagenTime


# evaluate only glucose sliding window dataset
ROOT=/mnt/unites8/playpen/haochenz/Drift/baselines/ImagenTime/ImagenTime
TMP=/tmp/imagentime_glucose_only

rm -rf "$TMP"
mkdir -p "$TMP"
ln -s "$ROOT"/GlucoseSliding_len* "$TMP"/

python eval_imagentime_discriminative_samples.py \
  --imagentime_root "$TMP" \
  --output_jsonl "$ROOT/imagentime_discriminative_glucose_results.jsonl" \
  --num_runs 10 \
  --config_root /playpen-shared/haochenz/Drift/baselines/ImagenFew/configs/ImagenTime
