#!/usr/bin/env bash
# R8: geometry-conditioned fused 3D residual, zero-init and H76-frozen.
set -euo pipefail

lr_variant=${1:?usage: $0 <lr1e5|lr1e4> <physical_gpu> [seed]}
physical_gpu=${2:?usage: $0 <lr1e5|lr1e4> <physical_gpu> [seed]}
seed=${3:-0}
case "${lr_variant}" in
  lr1e5) lr=1e-5 ;;
  lr1e4) lr=1e-4 ;;
  *) echo "unknown learning-rate variant: ${lr_variant}" >&2; exit 2 ;;
esac

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
prefix=${R8_TAG_PREFIX:-R8}
export CODE_OVERRIDE="${prefix}_${lr_variant}_seed${seed}"
export BASE_OVERRIDE="${ROOT}/R8_geometry_conditioned_output_20260809"
export TAG_OVERRIDE="${prefix}_H76_geometryConditionedOutputResidual_w322_${lr_variant}_seed${seed}_20260809"
export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; zero-init fused 3D residual conditioned on 4D ray-normal geometry; lr=${lr}; 5 epochs"
export SEED_OVERRIDE="${seed}"
export RUMPL_STACK_FROM=H76
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export H21_OVERRIDE="${ROOT}/H55_H58_h21_screen/H58_balanced_views_seed0/final.pth"

# Keep this as a single-variable ablation.
export RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0
export RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=1
export RUMPL_TRAIN_SCOPE=post_pft_geometry_conditional_residual
export RUMPL_TRI_ANCHOR=1
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_FINETUNE_LR="${lr}"
export RUMPL_END_EPOCH=5
export RUMPL_DISABLE_FIXED_VIEW_CURRICULUM=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export RUMPL_WORKERS="${RUMPL_WORKERS:-12}"

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
