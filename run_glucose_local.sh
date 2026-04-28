# calculate loss on feature domain
#GPU_ID=5 \
#EXP_NAME=glucose_TS2Vec_encoder \
#NUM_WORKERS=4 \
#USE_FEATURE_ENCODER=1 \
#WANDB_RUN_NAME="feature_loss" \
#TS_FEATURE_ENCODER_CKPT="/playpen-shared/haochenz/Drift/feature_extractors/checkpoints/full_series_ts2vec_glucose/best_full_series_ts2vec_glucose.pt" \
#bash train_glucose_ts_unconditional.sh

# calcualte loss on data domain
GPU_ID=5 \
EXP_NAME=glucose_no_ts_encoder \
NUM_WORKERS=4 \
USE_FEATURE_ENCODER=0 \
WANDB_RUN_NAME="data_loss" \
TEMPERATURES=0.02,0.05,0.2,0.5,0.8 \
bash train_glucose_ts_unconditional.sh
