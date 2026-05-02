CUDA_VISIBLE_DEVICES=4 \
REL_PATH=TSF/ETT-small/ETTm1.csv \
DATA_BACKEND=ETTm1 \
DATASET_NAME=ETTm1 \
IN_CHANNEL=7 \
ONE_CHANNEL=1 \
bash scripts/benchmark_drift.sh


CONDA_ENV=vlm \
REL_PATH=TSF/ETT-small/ETTm1.csv \
DATA_BACKEND=ETTm1 \
DATASET_NAME=ETTm1 \
IN_CHANNEL=7 \
ONE_CHANNEL=1 \
VAE_CKPT_NAME=last.pt \
sbatch scripts/benchmark_drift.sh



CONDA_ENV=vlm \
REL_PATH=TSF/ETT-small/ETTm2.csv \
DATA_BACKEND=ETTm2 \
DATASET_NAME=ETTm2 \
IN_CHANNEL=7 \
ONE_CHANNEL=1 \
VAE_CKPT_NAME=last.pt \
sbatch scripts/benchmark_drift.sh


CONDA_ENV=vlm \
REL_PATH=TSF/ETT-small/ETTh2.csv \
DATA_BACKEND=ETTh2 \
DATASET_NAME=ETTh2 \
IN_CHANNEL=7 \
ONE_CHANNEL=1 \
VAE_CKPT_NAME=last.pt \
sbatch scripts/benchmark_drift.sh


CONDA_ENV=vlm \
REL_PATH=TSF/weather/weather.csv \
DATA_BACKEND=custom \
DATASET_NAME=Weather \
IN_CHANNEL=21 \
ONE_CHANNEL=1 \
VAE_CKPT_NAME=last.pt \
sbatch scripts/benchmark_drift.sh


CONDA_ENV=vlm \
REL_PATH=TSG/AirQuality/AirQualityUCI.csv \
DATA_BACKEND=AirQuality \
DATASET_NAME=AirQuality \
IN_CHANNEL=13 \
ONE_CHANNEL=1 \
VAE_CKPT_NAME=last.pt \
sbatch scripts/benchmark_drift.sh
