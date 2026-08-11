#!/usr/bin/env bash
# R6: rerun geometry controls with a deterministic frozen H76 backbone.
set -euo pipefail

variant=${1:?usage: $0 <geometry_bias|geometry_token> <lr1e4|lr5e4> <physical_gpu> [seed]}
lr_variant=${2:?usage: $0 <geometry_bias|geometry_token> <lr1e4|lr5e4> <physical_gpu> [seed]}
physical_gpu=${3:?usage: $0 <geometry_bias|geometry_token> <lr1e4|lr5e4> <physical_gpu> [seed]}
seed=${4:-0}
case "${lr_variant}" in
  lr1e4) lr=1e-4 ;;
  lr5e4) lr=5e-4 ;;
  *) echo "unknown learning-rate variant: ${lr_variant}" >&2; exit 2 ;;
esac

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
BASE=${ROOT}/R6_deterministic_frozen_controls_20260809

export CODE_OVERRIDE="R6_${variant}_${lr_variant}_seed${seed}"
export BASE_OVERRIDE="${BASE}"
export TAG_OVERRIDE="R6_H76_deterministic_${variant}_w322_${lr_variant}_seed${seed}_20260809"
export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; geometry ${variant}; only target module train; lr=${lr}; 5 epochs"
export SEED_OVERRIDE="${seed}"
export RUMPL_STACK_FROM=H76
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export H21_OVERRIDE="${ROOT}/H55_H58_h21_screen/H58_balanced_views_seed0/final.pth"

# Explicitly disable every optional branch before enabling exactly one control.
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
export RUMPL_TRAIN_SCOPE=all
case "${variant}" in
  geometry_bias)
    export CODE_OVERRIDE="R6a_${lr_variant}_seed${seed}"
    export TAG_OVERRIDE="R6a_H76_deterministic_geometryViewBias_w322_${lr_variant}_seed${seed}_20260809"
    export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; R2b geometry view-bias only; lr=${lr}; 5 epochs"
    export RUMPL_GEOMETRY_VIEW_BIAS=1
    export RUMPL_TRAIN_SCOPE=geometry_view_bias
    ;;
  geometry_token)
    export CODE_OVERRIDE="R6b_${lr_variant}_seed${seed}"
    export TAG_OVERRIDE="R6b_H76_deterministic_geometryTokenResidual_w322_${lr_variant}_seed${seed}_20260809"
    export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; R4b geometry token residual only; lr=${lr}; 5 epochs"
    export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=1
    export RUMPL_TRAIN_SCOPE=joint_geometry_token_residual
    ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac

export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_FINETUNE_LR="${lr}"
export RUMPL_END_EPOCH=5
export RUMPL_DISABLE_FIXED_VIEW_CURRICULUM=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export RUMPL_WORKERS="${RUMPL_WORKERS:-12}"

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
