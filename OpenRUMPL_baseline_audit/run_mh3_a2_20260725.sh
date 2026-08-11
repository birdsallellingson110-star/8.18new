#!/usr/bin/env bash
# MH3 = A2 + MHFormer-style H=3 fusion tokens (collapse before PFT; loss on final only).
# usage: $0 GPU [VARIANT]
set -euo pipefail

gpu=${1:?usage: $0 GPU [VARIANT]}
variant=${2:-MH3_a2_seed0_20260725}

unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0

export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_ADAFUSE_VW=0 RUMPL_2D_REFINE=0 RUMPL_POSE_CODEBOOK=0
export RUMPL_KPA=0 RUMPL_CONF_FILM=0

# A2 conditions
export RUMPL_TRAIN_STRUCT_OCC=1
export RUMPL_TRAIN_STRUCT_OCC_LEVEL=0.4
export RUMPL_OCC_JOINT_LOSS=1
export RUMPL_OCC_JOINT_LOSS_BOOST=2.0
export RUMPL_OCC_JOINT_LOSS_MODE=soft

# MHFormer H=3
export RUMPL_MULTI_HYP=3

echo "MH3 fusion tokens + A2 variant=$variant"

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
