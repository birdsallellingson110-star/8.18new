#!/usr/bin/env bash
# 用法: bash run_step2_split.sh <split_number> <gpu_id> [subset=train]
# 单个 stage_IV split → stage_V (渲染 + HRNet + 三角化)。每个 split 独立保存 pkl。
set -euo pipefail
SPLIT=$1
GPU=$2
SUBSET=${3:-train}

source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1
# HRNet 权重走师弟缓存，避免联网下载
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export CUDA_VISIBLE_DEVICES=$GPU
# 限制每进程 BLAS 线程，防止 80 核被多进程超订 (8 workers x 8 ≈ 64 线程)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
# 减少显存碎片，降低 OOM 风险
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=/mnt/data/cjydata
cd /home/lixiaob/cjy/OpenRUMPL/MHP

python -u run_mmpose_02_run.py \
  --dataset-split-number "$SPLIT" \
  --exp paper_single_cmu \
  --extra-name random_20_small_room_person_dist_2 \
  --views-cmu 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
  --use-cams-from cmu \
  --calib-root-cmu "$DATA/cmu_calibs" \
  --calibs-cmu 171204_pose5 171204_pose6 \
  --room-size -0.5 -0.1 -0.2 0.2 0 0 \
  --operation-on "$SUBSET" \
  --image-width 1920 --image-height 1080 \
  --apply-rotation \
  --regressor coco \
  --triangulate --triangulate-th 0.95 \
  --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 \
  --save-temp-checkpoints \
  --run-on-random-cameras \
  --n-cameras-per-person 20 \
  --camera-location-limit -2.7 2.7 -2.7 2.7 0.7 3.4 \
  --camera-dist-from-person 2 \
  --image-save-dir "$DATA/mhp_images" \
  --support-dir /mnt/data/dataset/c2i/rumpl_support \
  --work-dir "$DATA/mhp_workspace" \
  --amass-data-dir /mnt/data/dataset/c2i/AMASS_smpl_c2i
