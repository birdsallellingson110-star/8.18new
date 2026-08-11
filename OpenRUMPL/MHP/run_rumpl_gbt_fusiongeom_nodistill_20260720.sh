#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {geom|both} {0.1|1.0} [physical_gpu]" >&2
  exit 2
fi

mode=$1
geom_init=$2
gpu=${3:-0}
case "$mode" in
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

case "$geom_init" in
  0.1) init_tag=g01 ;;
  1.0) init_tag=g1 ;;
  *)
    echo "Unsupported geometry initialization: $geom_init" >&2
    exit 2
    ;;
esac

cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1

export CUDA_VISIBLE_DEVICES="$gpu"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache
export PIP_CACHE_DIR=/mnt/data/cjydata/.pip_cache
export WANDB_MODE=offline

export DISTILL_W=0 DISTILL_LAMBDA=0 FEAT_DISTILL_W=0 STUDENT_GT_W=0
export HARD_VIEW_MINING=0 HARD_VIEW_CAND=0 LEG_DISTILL_W=1 AUX_MULTIK_W=0 RCG=0

export GBT_LEARNABLE_BIAS=1
export GBT_USE_CONF_BIAS=$use_conf
export GBT_USE_GEOM_BIAS=$use_geom
export GBT_CONF_INIT=0.1
export GBT_GEOM_INIT="$geom_init"
export GBT_FUSION_GEOM=1
export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0
unset GBT_V2_SCALE GBT_V3_SCALE GBT_V4_SCALE

export GBT_TOKEN_DROPOUT=0 VFT_FULL_RANDOM_MASK=0
export REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
export RAY_LAMBDA=0 STUDENT_RAY_W=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0
export BONE_LAMBDA=0

exec /home/lixiaob/cjy/rumpl_venv310/bin/python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml \
  --gpus 0 \
  --workers 4 \
  --exp-name "rumpl_gbt_fusiongeom_nodistill_${mode}_${init_tag}_20260720"
