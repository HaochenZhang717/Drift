python eval_soft_vqvae_recon_neighbors.py \
  --ckpt /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/ErcotData/best.pt \
  --output_dir /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/ErcotData/eval_val \
  --device cuda \
  --max_val_samples 2000 \
  --num_plot 12


python eval_soft_vqvae_recon_neighbors.py \
  --ckpt /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/HouseholdData/best.pt \
  --output_dir /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/HouseholdData/eval_val \
  --device cuda \
  --max_val_samples 2000 \
  --num_plot 12

python eval_soft_vqvae_recon_neighbors.py \
  --ckpt /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/GlucoseSliding/best.pt \
  --output_dir /mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark/GlucoseSliding/eval_val \
  --device cuda \
  --max_val_samples 2000 \
  --num_plot 12
