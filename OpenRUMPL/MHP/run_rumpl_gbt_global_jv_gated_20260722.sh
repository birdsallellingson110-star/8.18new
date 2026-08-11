#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {nodrop|stage5drop} PHYSICAL_GPU" >&2
  exit 2
fi

mode=$1
gpu=$2
case "$mode" in
  nodrop) token_dropout=0; token_dropout_epochs=0 ;;
  stage5drop) token_dropout=0.1; token_dropout_epochs=5 ;;
  *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
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
export HARD_VIEW_MINING=0 HARD_VIEW_CAND=0 AUX_MULTIK_W=0 RCG=0
export MONO_W=0 MONO_GT_W=0 MONO_MARGIN=0
export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=5

export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=1.0 GBT_FUSION_GEOM=0
export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0
unset GBT_V2_SCALE GBT_V3_SCALE GBT_V4_SCALE

export GBT_GLOBAL_JV_DEPTH=1 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=1
export GBT_TOKEN_DROPOUT="$token_dropout"
export GBT_TOKEN_DROPOUT_EPOCHS="$token_dropout_epochs"
export VFT_FULL_RANDOM_MASK=0
export REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
export RAY_LAMBDA=0 STUDENT_RAY_W=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0
export BONE_LAMBDA=0

echo "[GLOBAL_JV_GATED] mode=$mode gpu=$gpu gate_init=0 token_dropout=$token_dropout token_dropout_epochs=$token_dropout_epochs"
exec /home/lixiaob/cjy/rumpl_venv310/bin/python run/train_rumpl.py \
  --cfg configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml \
  --gpus 0 \
  --workers 4 \
  --seed 0 \
  --exp-name "rumpl_gbt_global_jv_gated_${mode}_20260722"
