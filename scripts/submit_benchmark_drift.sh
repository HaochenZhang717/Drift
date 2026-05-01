#CUDA_VISIBLE_DEVICES=1 \
#REL_PATH=TSF/ETT-small/ETTm1.csv \
#DATA_BACKEND=ETTm1 \
#DATASET_NAME=ETTm1 \
#IN_CHANNEL=7 \
#bash scripts/benchmark_drift.sh


CUDA_ENV=vlm \
REL_PATH=TSF/ETT-small/ETTm1.csv \
DATA_BACKEND=ETTm1 \
DATASET_NAME=ETTm1 \
IN_CHANNEL=7 \
sbatch scripts/benchmark_drift.sh