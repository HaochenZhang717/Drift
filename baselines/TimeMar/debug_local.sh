DATANAMES=("stock" "ETTh1" "fmri" "energy" "ETTh2" "ETTm1" "ETTm2"  )
#DATANAMES=("stock" )

VQVAEDIR="../dual_vqvae_save_dir_0207"
VARDIR="../var_save_dir_0207"
for DATA in "${DATANAMES[@]}"
do
  VQVAECONFIG="configs/train_vq_${DATA}.yaml"
  VARCONFIG="configs/train_var_${DATA}.yaml"
  VQVAECKPT="${VQVAEDIR}/vq_${DATA}/checkpoints/latest.pt"

  python train_dual_vqvae.py \
  --data ${DATA} \
  --config  ${VQVAECONFIG} \
  --max_epochs 0 \
  --val_every 100 \
  --save_dir ${VQVAEDIR}

#  CUDA_VISIBLE_DEVICES=0 python train_ar.py \
#  --data ${DATA} \
#  --vqvae_path ${VQVAECKPT} \
#  --config  ${VARCONFIG} \
#  --save_dir ${VARDIR}
done



