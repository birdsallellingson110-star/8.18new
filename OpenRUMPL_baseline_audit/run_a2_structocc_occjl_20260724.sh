#!/usr/bin/env bash
# A2 = S2a (structured limb occlusion) + occluded-joint weighted MPJPE.
# Single new variable vs S2a: RUMPL_OCC_JOINT_LOSS=1.
# usage: $0 GPU [VARIANT] [STRUCT_OCC_LEVEL] [BOOST]
set -euo pipefail

gpu=${1:?usage: $0 GPU [VARIANT] [STRUCT_OCC_LEVEL] [BOOST]}
variant=${2:-A2_structocc_occjl_b2_seed0_20260724}
level=${3:-0.4}
boost=${4:-2.0}

# distill OFF
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0

# R5 protocol extras OFF
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0

# S2a: structured limb occlusion (same as Stage2a)
export RUMPL_TRAIN_STRUCT_OCC=1
export RUMPL_TRAIN_STRUCT_OCC_LEVEL="$level"

# A2: occluded-joint weighted loss (NEW vs S2a)
export RUMPL_OCC_JOINT_LOSS=1
export RUMPL_OCC_JOINT_LOSS_BOOST="$boost"
export RUMPL_OCC_JOINT_LOSS_MODE=soft

echo "A2 variant=$variant STRUCT_OCC=$level OCC_JOINT_LOSS boost=$boost mode=soft (distill OFF)"

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
