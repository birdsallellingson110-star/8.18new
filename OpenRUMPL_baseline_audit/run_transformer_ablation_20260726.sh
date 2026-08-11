#!/usr/bin/env bash
# Transformer structure ablations on the clean official R5 training protocol.
# usage: $0 GPU MODE [VARIANT]
# MODE: t0_fix | t1_shallow | t2_alt
set -euo pipefail

gpu=${1:?usage: $0 GPU MODE [VARIANT]}
mode=${2:?usage: $0 GPU MODE [VARIANT]}
variant=${3:-${mode}_a2_seed0_20260726}

# Disable all unrelated experimental branches.
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W \
  HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_ADAFUSE_VW=0 RUMPL_2D_REFINE=0
export RUMPL_POSE_CODEBOOK=0 RUMPL_KPA=0 RUMPL_CONF_FILM=0
export RUMPL_MULTI_HYP=1

# R5 protocol: no structured occlusion or occlusion-weighted loss.
export RUMPL_TRAIN_STRUCT_OCC=0
export RUMPL_TRAIN_STRUCT_OCC_LEVEL=0
export RUMPL_OCC_JOINT_LOSS=0
export RUMPL_OCC_JOINT_LOSS_BOOST=1.0
export RUMPL_OCC_JOINT_LOSS_MODE=soft

case "$mode" in
  t0_fix)
    # Original 12/12 stacks, but execute the final PFT block only once.
    export RUMPL_ALT_JOINT_VIEW=0
    export RUMPL_VFT_DEPTH=12
    export RUMPL_PFT_DEPTH=12
    ;;
  t1_shallow)
    # Twelve total blocks: test whether V=2..5 was over-processed.
    export RUMPL_ALT_JOINT_VIEW=0
    export RUMPL_VFT_DEPTH=4
    export RUMPL_PFT_DEPTH=8
    ;;
  t2_alt)
    # 4*(view+joint) + one view readout + 4 PFT = 13 blocks.
    export RUMPL_ALT_JOINT_VIEW=1
    export RUMPL_ALT_JOINT_VIEW_DEPTH=4
    export RUMPL_VFT_DEPTH=0
    export RUMPL_PFT_DEPTH=4
    ;;
  *)
    echo "unknown mode: $mode (want t0_fix|t1_shallow|t2_alt)" >&2
    exit 2
    ;;
esac

echo "TRANSFORMER mode=$mode variant=$variant alt=$RUMPL_ALT_JOINT_VIEW vft=$RUMPL_VFT_DEPTH pft=$RUMPL_PFT_DEPTH"

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 1 1 16
