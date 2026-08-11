#!/usr/bin/env bash
# Train RUMPL with full-view -> sparse-view self-distillation on strict R5 protocol.
# usage: $0 GPU VARIANT MODE
#   MODE: general | hardv_legw09 | hardv_legw07
set -euo pipefail

gpu=${1:?usage: $0 GPU VARIANT MODE}
variant=${2:?usage: $0 GPU VARIANT MODE}
mode=${3:?usage: $0 GPU VARIANT MODE}

export DISTILL_W=1
export STUDENT_GT_W=1
export FEAT_DISTILL_W=0
export STUDENT_VIEWS=2
export HARD_VIEW_MINING=0
export HARD_VIEW_CAND=3
export LEG_DISTILL_W=1.0

case "$mode" in
  general)
    ;;
  hardv_legw09)
    export HARD_VIEW_MINING=1
    export LEG_DISTILL_W=0.9
    ;;
  hardv_legw07)
    export HARD_VIEW_MINING=1
    export LEG_DISTILL_W=0.7
    ;;
  *)
    echo "unknown mode: $mode (use general | hardv_legw09 | hardv_legw07)" >&2
    exit 2
    ;;
esac

echo "DISTILL_W=$DISTILL_W STUDENT_GT_W=$STUDENT_GT_W STUDENT_VIEWS=$STUDENT_VIEWS"
echo "HARD_VIEW_MINING=$HARD_VIEW_MINING LEG_DISTILL_W=$LEG_DISTILL_W"

# R5 protocol: public PFT, fixed scheduler, 16 workers, no GBT/extra modules.
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
