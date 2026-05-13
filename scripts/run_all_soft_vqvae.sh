export CUDA_VISIBLE_DEVICES=7

TS_SEQ_LEN=512 \
DELAY=8 \
EMBEDDING=64 \
TS_DATA_DIR=/mnt/unites8/playpen/haochenz/Time_Series_Datasets \
SAVE_ROOT=/mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark \
bash scripts/train_soft_vqvae_ercot.sh



TS_SEQ_LEN=512 \
DELAY=8 \
EMBEDDING=64 \
TS_DATA_DIR=/playpen-shared/haochenz/AI-READI-Dataset/AI-READI-processed \
SAVE_ROOT=/mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark \
bash scripts/train_soft_vqvae_glucosesliding.sh


TS_SEQ_LEN=512 \
DELAY=8 \
EMBEDDING=64 \
TS_DATA_DIR=/mnt/unites8/playpen/haochenz/Time_Series_Datasets \
SAVE_ROOT=/mnt/unites8/playpen/haochenz/Drift/soft_vqvae_benchmark \
bash scripts/train_soft_vqvae_household.sh
