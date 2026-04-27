GPU_ID=6 \
LR_LIST="1e-3 2e-4 5e-5" \
BATCH_SIZE_LIST="256 512 1024" \
EPOCHS=50 \
WANDB_PROJECT=drifting-model-ts \
bash train_full_series_ts2vec_glucose_sweep_lr_bs.sh
