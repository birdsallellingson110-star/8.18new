#!/usr/bin/env bash
# Stage 2a: R5 protocol + structured limb-group occlusion augmentation (data-only).
# Single-variable change vs R5: RUMPL_TRAIN_STRUCT_OCC=1. No distillation, no GBT, no extra modules.
# usage: $0 GPU [VARIANT] [STRUCT_OCC_LEVEL]
set -euo pipefail

gpu=${1:?usage: $0 GPU [VARIANT] [STRUCT_OCC_LEVEL]}
variant=${2:-S2a_structocc_L04_seed0_20260724}
level=${3:-0.4}

# --- ensure distillation is fully OFF (it triggers on any of these) ---
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0

# --- R5-exact protocol: public PFT, fixed scheduler, no GBT / anchor / depth / global / symmetry ---
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0

# --- the ONLY new ingredient: structured limb-group occlusion during training ---
export RUMPL_TRAIN_STRUCT_OCC=1
export RUMPL_TRAIN_STRUCT_OCC_LEVEL="$level"

echo "STAGE2a variant=$variant RUMPL_TRAIN_STRUCT_OCC=1 LEVEL=$level (distill OFF)"

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
