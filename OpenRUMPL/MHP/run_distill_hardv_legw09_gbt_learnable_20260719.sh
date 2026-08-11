#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {conf|geom|both}" >&2
  exit 2
fi

mode=$1
case "$mode" in
  conf)
    use_conf=1
    use_geom=0
    ;;
  geom)
    use_conf=0
    use_geom=1
    ;;
  both)
    use_conf=1
    use_geom=1
    ;;
  *)
    echo "Unsupported mode: $mode" >&2
    exit 2
    ;;
esac

cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1

export CUDA_VISIBLE_DEVICES=1
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache
export PIP_CACHE_DIR=/mnt/data/cjydata/.pip_cache
export WANDB_MODE=offline

# Verified Distill-RUMPL baseline. No auxiliary multi-K or K-specific geometry rule.
export DISTILL_W=1 STUDENT_GT_W=1 STUDENT_VIEWS=2
export HARD_VIEW_MINING=1 HARD_VIEW_CAND=3 LEG_DISTILL_W=0.9
export AUX_MULTIK_W=0

# Paper-faithful unified biased attention shared by every input view count.
export GBT_LEARNABLE_BIAS=1
export GBT_USE_CONF_BIAS=$use_conf
export GBT_USE_GEOM_BIAS=$use_geom
export GBT_CONF_INIT=0.1
export GBT_GEOM_INIT=1.0
export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0
unset GBT_V2_SCALE GBT_V3_SCALE GBT_V4_SCALE

export GBT_TOKEN_DROPOUT=0 VFT_FULL_RANDOM_MASK=0
export REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
export RAY_LAMBDA=0 STUDENT_RAY_W=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0

exec /home/lixiaob/cjy/rumpl_venv310/bin/python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml \
  --gpus 0 \
  --workers 8 \
  --exp-name "distill_hardv_legw09_gbt_learnable_${mode}_20260719"
