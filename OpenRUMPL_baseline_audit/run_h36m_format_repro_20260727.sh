#!/usr/bin/env bash
set -euo pipefail

# Stage A of the Human3.6M reproduction:
# train the clean RUMPL baseline on the already generated AMASS data expressed
# in the H36M-17 joint convention.  This validates the model/data pipeline,
# but it is not the paper-number test on real Human3.6M S9/S11.

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=/home/lixiaob/cjy/OpenRUMPL/RUMPL/configs/h36m/rumpl_amass/clip_full_h36m_flat_conf.yaml
run_root=/mnt/data/cjyoutput/h36m_repro_20260727
variant=h36m17_format_rumpl_clean_seed0_20260727
log="$run_root/${variant}.log"

mkdir -p "$run_root"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled

# Match the verified R5 baseline recipe.  All optional transformer/bias
# branches stay disabled by default.
export RUMPL_FIX_PFT_LAST_BLOCK=0
export RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=0
export GBT_FUSION_GEOM=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_SINGLEFRAME_GBT=0
export RUMPL_JOINT_ADAPTER_DIRECT_READOUT=0
export RUMPL_TRAIN_STRUCT_OCC=0
export RUMPL_OCC_JOINT_LOSS=0
export RUMPL_KPA=0
export RUMPL_MULTI_HYP=0
export RUMPL_POSE_CODEBOOK=0

exec > >(tee -a "$log") 2>&1
trap 'rc=$?; echo "FAILED line=$LINENO rc=$rc $(date --iso-8601=seconds)"; exit "$rc"' ERR

echo "START $(date --iso-8601=seconds)"
echo "STAGE H36M-17 format pipeline; synthetic AMASS train/validation"
echo "PAPER_TARGET real H36M S9/S11 V2 All-17=52.5mm KP*=56.8mm"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONFIG=$cfg"
echo "DATASET=/mnt/data/cjydata/mhp_workspace/clip_full_h36m_flat/datasets/official_combined_h36m_flat"

cd "$repo/RUMPL"
"$python" run/train_rumpl.py \
  --cfg "$cfg" \
  --gpus 0 \
  --workers 4 \
  --exp-name "$variant"

echo "END $(date --iso-8601=seconds)"
