#!/usr/bin/env bash
set -euo pipefail
gpu=$1; variant=$2; workers=${3:-8}
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_OCC_JOINT_LOSS=0 RUMPL_ADAFUSE_VW=0
export RUMPL_TRAIN_STRUCT_OCC=1 RUMPL_TRAIN_STRUCT_OCC_LEVEL=0.4
# train-time soft 2D/ray refine (paper-inspired consensus fill)
export RUMPL_2D_REFINE=1
export RUMPL_2D_REFINE_MODE=soft_fill
export RUMPL_2D_REFINE_STRENGTH=0.3
export RUMPL_2D_REFINE_FILL_CONF=0.35
export RUMPL_2D_REFINE_CONF_THR=0.1
echo "2D-refine TRAIN soft_fill s=0.3 + struct_occ=0.4 variant=$variant"
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 "$workers"
