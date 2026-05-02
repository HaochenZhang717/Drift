# One Channel
python train_fid_vae_benchmark.py \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/ETT-small/ETTm1.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
  --one_channel


python train_fid_vae_benchmark.py \
  --dataset_name "ETTm2" \
  --data "ETTm2" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/ETT-small/ETTm2.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
  --one_channel



python train_fid_vae_benchmark.py \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/ETT-small/ETTh2.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
  --one_channel

python train_fid_vae_benchmark.py \
  --dataset_name "Weather" \
  --data "custom" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/weather/weather.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
  --one_channel

python train_fid_vae_benchmark.py \
  --dataset_name "AirQuality" \
  --data "AirQuality" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSG/AirQuality/AirQualityUCI.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
  --one_channel

# Multi Channel
python train_fid_vae_benchmark.py \
  --dataset_name "ETTm1" \
  --data "ETTm1" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/ETT-small/ETTm1.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \


python train_fid_vae_benchmark.py \
  --dataset_name "ETTm2" \
  --data "ETTm2" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/ETT-small/ETTm2.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \



python train_fid_vae_benchmark.py \
  --dataset_name "ETTh2" \
  --data "ETTh2" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/ETT-small/ETTh2.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \


python train_fid_vae_benchmark.py \
  --dataset_name "Weather" \
  --data "custom" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSF/weather/weather.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
  --one_channel


python train_fid_vae_benchmark.py \
  --dataset_name "AirQuality" \
  --data "AirQuality" \
  --datasets_dir "/playpen-shared/haochenz/ImagenFew/data" \
  --rel_path "TSG/AirQuality/AirQualityUCI.csv" \
  --ts_seq_len 256 \
  --batch_size 128 \
  --epochs 100 \
  --save_dir ./fid_vae_ckpts/benchmark_256 \
