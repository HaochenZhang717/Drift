#!/bin/bash
#SBATCH --job-name=test-conda
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=test-conda-%j.out

echo "===== JOB INFO ====="
hostname
nvidia-smi

echo "===== CHECK PLAYPEN PATH ACCESS ====="
check_path_access() {
  local p="$1"
  if [ ! -e "$p" ]; then
    echo "[MISS] $p (does not exist)"
    return
  fi

  local r="-"
  local x="-"
  local list="-"
  [ -r "$p" ] && r="r"
  [ -x "$p" ] && x="x"
  if ls -ld "$p" >/dev/null 2>&1; then
    list="ok"
  fi

  if [ "$r" = "r" ] && [ "$x" = "x" ] && [ "$list" = "ok" ]; then
    echo "[PASS] $p (read/list accessible)"
  else
    echo "[FAIL] $p (r=$r x=$x list=$list)"
  fi
}

for i in 2 3 4 5 6 7 8 9; do
  check_path_access "/mnt/unites${i}/playpen/haochenz"
done
check_path_access "/playpen/haochenz"

echo "===== LOAD CONDA ====="
# 清掉可能污染的 conda
unset -f conda __conda_activate __conda_exe __conda_hashr
unset CONDA_EXE CONDA_PREFIX CONDA_SHLVL _CE_CONDA _CE_M
hash -r

# 加载你自己的 conda
#source /mnt/unites9/playpen/haochenz/miniconda3/etc/profile.d/conda.sh
source /playpen/haochenz/miniconda3/etc/profile.d/conda.sh

echo "===== ACTIVATE ENV ====="
conda activate qwen3_vl_eval

echo "===== CHECK PATH ====="
echo "conda: $(which conda)"
echo "python: $(which python)"

echo "===== CHECK TORCH ====="
python - << 'EOF'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
EOF

echo "===== DONE ====="
