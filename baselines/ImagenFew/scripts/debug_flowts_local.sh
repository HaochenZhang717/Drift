OUTPUT_ROOT="/Users/zhc/Documents/PhD/projects/drifting-model/baselines/ImagenFew/debug_FlowTS"

HOSTNAME_SHORT="$(hostname -s)"

echo "Host=${HOSTNAME_SHORT}, using OUTPUT_ROOT=${OUTPUT_ROOT}"

CONFIG_FILE="/Users/zhc/Documents/PhD/projects/drifting-model/baselines/ImagenFew/configs/FlowTS/debug_ErcotData_len256.yaml"

python /Users/zhc/Documents/PhD/projects/drifting-model/baselines/ImagenFew/run.py \
--subset_p 1.0 \
--log_dir "${OUTPUT_ROOT}" \
--config ${CONFIG_FILE} \
--epochs 3 \
--logging_iter 1 \
--no_test_model
