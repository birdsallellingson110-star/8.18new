#!/usr/bin/env bash
# B1 / AdaFuse ViewWeightNet (ray-space) + structured limb occlusion training.
# Faithful transferable core of AdaFuse (IJCV'21):
#   pairwise geometry consistency + conf → learned per-(joint,view) weight
#   → scale view tokens before VFT (occluded views → ~0).
# Heatmap branch dropped (RUMPL lifting has no heatmaps).
# usage: $0 GPU [VARIANT] [STRUCT_OCC_LEVEL]
set -euo pipefail

gpu=${1:?usage: $0 GPU [VARIANT] [STRUCT_OCC_LEVEL]}
variant=${2:-B1_adafuse_vw_structocc04_seed0_20260725}
level=${3:-0.4}

unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0

export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_OCC_JOINT_LOSS=0
export RUMPL_2D_REFINE=0

# training condition (same as S2a / AdaFuse needs occlusion to learn reliability)
export RUMPL_TRAIN_STRUCT_OCC=1
export RUMPL_TRAIN_STRUCT_OCC_LEVEL="$level"

# AdaFuse view weight (NEW)
export RUMPL_ADAFUSE_VW=1
export RUMPL_ADAFUSE_VW_MIX=0.0

echo "B1 AdaFuse-VW variant=$variant STRUCT_OCC=$level (distill/occjl/2drefine OFF)"

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
