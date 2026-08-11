#!/usr/bin/env bash
set -euo pipefail
cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=1
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache PIP_CACHE_DIR=/mnt/data/cjydata/.pip_cache
export WANDB_MODE=offline
export DISTILL_W=1 STUDENT_GT_W=1 STUDENT_VIEWS=2
export HARD_VIEW_MINING=1 HARD_VIEW_CAND=3 LEG_DISTILL_W=0.9
export AUX_MULTIK_W=0.5 AUX_MULTIK_MIN=3 AUX_MULTIK_MAX=4
export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0.12 GBT_VIEW_AWARE=1
export GBT_V2_SCALE=0 GBT_V3_SCALE=1 GBT_V4_SCALE=1 GBT_TOKEN_DROPOUT=0
export VFT_FULL_RANDOM_MASK=0 REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
export RAY_LAMBDA=0 STUDENT_RAY_W=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0
exec /home/lixiaob/cjy/rumpl_venv310/bin/python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml --gpus 0 --workers 8 \
  --exp-name distill_dualhard_legw09_gbt_k3plus_aux05_20260718
