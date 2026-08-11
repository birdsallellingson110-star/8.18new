#!/usr/bin/env bash
set -euo pipefail

cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1

export CUDA_VISIBLE_DEVICES=1
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache
export PIP_CACHE_DIR=/mnt/data/cjydata/.pip_cache
export WANDB_MODE=offline

export DISTILL_W=1
export STUDENT_GT_W=1
export STUDENT_VIEWS=2
export HARD_VIEW_MINING=1
export HARD_VIEW_CAND=3
export LEG_DISTILL_W=0.9

# The 0.2 run improved mean V2/V3 but over-regularized weak geometry.
export REPROJ_LAMBDA=0.0
export STUDENT_REPROJ_W=0.1
export REPROJ_BETA=0.01
export RAY_LAMBDA=0.0
export STUDENT_RAY_W=0.0
export VFT_FULL_RANDOM_MASK=0.0
export GBT_CONF_BIAS=0.0
export GBT_GEOM_BIAS=0.0
export CAA_LAMBDA=0.0
export DEPRO_LAMBDA=0.0

exec /home/lixiaob/cjy/rumpl_venv310/bin/python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml \
  --gpus 0 \
  --workers 8 \
  --exp-name distill_hardv_legw09_reproj_student01_20260717
