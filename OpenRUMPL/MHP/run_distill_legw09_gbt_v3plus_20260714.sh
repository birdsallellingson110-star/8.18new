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

# V3+ reliability experiment:
# random sparse student views instead of hard V=2 mining, so the model learns
# a broader view-number distribution. V=2 geometry bias is disabled.
export DISTILL_W=1
export STUDENT_GT_W=1
export STUDENT_VIEWS=rand
export HARD_VIEW_MINING=0
export HARD_VIEW_CAND=3
export LEG_DISTILL_W=0.9

# Geometry-only bias. Keep confidence bias off after the first GBT-v2 run hurt [3,12].
export GBT_CONF_BIAS=0.0
export GBT_GEOM_BIAS=0.12
export GBT_VIEW_AWARE=1
export GBT_V2_SCALE=0.0
export GBT_TOKEN_DROPOUT=0.05

exec /home/lixiaob/cjy/rumpl_venv310/bin/python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml \
  --gpus 0 \
  --workers 8 \
  --exp-name distill_legw09_gbt_v3plus_20260714
