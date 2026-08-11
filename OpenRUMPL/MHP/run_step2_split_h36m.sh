#!/usr/bin/env bash
# Strict H36M single-frame MHP generation.
# Reuses paper_single_cmu/stage_IV splits, but reruns render+HRNet with the
# H36M joint regressor and H36M camera geometry.
#
# Usage: bash run_step2_split_h36m.sh <split_number> <gpu_id> [subset=train]
set -euo pipefail

SPLIT=$1
GPU=$2
SUBSET=${3:-train}

source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=/mnt/data/cjydata
WORK="$DATA/mhp_workspace"
EXP=paper_single_h36m
SRC_EXP=paper_single_cmu

mkdir -p "$WORK/$EXP"
if [ ! -e "$WORK/$EXP/stage_IV" ]; then
  ln -s "$WORK/$SRC_EXP/stage_IV" "$WORK/$EXP/stage_IV"
fi

cd /home/lixiaob/cjy/OpenRUMPL/MHP

python -u run_mmpose_02_run.py \
  --dataset-split-number "$SPLIT" \
  --exp "$EXP" \
  --extra-name random_20_small_room_h36m \
  --use-cams-from h36m \
  --calib-file-h36m /mnt/data/dataset/c2i/camera_data.pkl \
  --actors-h36m 9 11 \
  --room-size -1 1 -1.5 2 0 0 \
  --operation-on "$SUBSET" \
  --image-width 1000 --image-height 1000 \
  --apply-rotation \
  --regressor h36m \
  --triangulate --triangulate-th 0.95 \
  --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 \
  --save-temp-checkpoints \
  --run-on-random-cameras \
  --n-cameras-per-person 20 \
  --camera-location-limit -2.2 2.2 -5.2 5.2 1 2 \
  --image-save-dir "$DATA/mhp_images_h36m" \
  --support-dir /mnt/data/dataset/c2i/rumpl_support \
  --work-dir "$WORK" \
  --amass-data-dir /mnt/data/dataset/c2i/AMASS_smpl_c2i
