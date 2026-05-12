DATANAMES=("synth_m")


for DATA in "${DATANAMES[@]}"
do
  VQVAEDIR="./dual_vqvae_save_dir_${DATA}"
  VARDIR="./var_save_dir_${DATA}/var_${DATA}"
  VQVAECONFIG="configs/neurips_baselines/train_vq_${DATA}.yaml"
  VARCONFIG="configs/neurips_baselines/train_var_${DATA}.yaml"
  VQVAECKPT="${VQVAEDIR}/vq_${DATA}/checkpoints/best.pt"

  python train_dual_vqvae.py \
    --data ${DATA} \
    --config ${VQVAECONFIG} \
    --max_epochs 5000 \
    --val_every 100 \
    --save_dir ${VQVAEDIR}

  python train_ar.py \
    --data ${DATA} \
    --vqvae_path ${VQVAECKPT} \
    --config ${VARCONFIG} \
    --save_dir ${VARDIR}
done