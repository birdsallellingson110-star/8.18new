#!/usr/bin/env bash
# R7: frozen-H76 output-side controls, to be launched after R6 results.
set -euo pipefail

variant=${1:?usage: $0 <residual_gate|joint_head|graph_residual> <lr1e4|lr5e4> <physical_gpu> [seed]}
lr_variant=${2:?usage: $0 <residual_gate|joint_head|graph_residual> <lr1e4|lr5e4> <physical_gpu> [seed]}
physical_gpu=${3:?usage: $0 <residual_gate|joint_head|graph_residual> <lr1e4|lr5e4> <physical_gpu> [seed]}
seed=${4:-0}
case "${lr_variant}" in
  lr1e5) lr=1e-5 ;;
  lr1e4) lr=1e-4 ;;
  lr5e4) lr=5e-4 ;;
  *) echo "unknown learning-rate variant: ${lr_variant}" >&2; exit 2 ;;
esac

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
BASE=${ROOT}/R7_output_side_frozen_20260809

export CODE_OVERRIDE="R7_${variant}_${lr_variant}_seed${seed}"
export BASE_OVERRIDE="${BASE}"
export TAG_OVERRIDE="R7_H76_outputSide_${variant}_w322_${lr_variant}_seed${seed}_20260809"
export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; output-side ${variant}; only target module train; lr=${lr}; 5 epochs"
export SEED_OVERRIDE="${seed}"
export RUMPL_STACK_FROM=H76
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export H21_OVERRIDE="${ROOT}/H55_H58_h21_screen/H58_balanced_views_seed0/final.pth"

# Clear every optional branch, then enable exactly one output-side module.
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
  residual_gate)
    export CODE_OVERRIDE="R7a_${lr_variant}_seed${seed}"
    export TAG_OVERRIDE="R7a_H76_outputSide_residualGate_w322_${lr_variant}_seed${seed}_20260809"
    export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; per-joint residual gate beside tri-anchor; lr=${lr}; 5 epochs"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_TRAIN_SCOPE=per_joint_residual_gate
    ;;
  joint_head)
    export CODE_OVERRIDE="R7b_${lr_variant}_seed${seed}"
    export TAG_OVERRIDE="R7b_H76_outputSide_jointSpecificHead_w322_${lr_variant}_seed${seed}_20260809"
    export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; zero-init per-joint 3D head; lr=${lr}; 5 epochs"
    export RUMPL_JOINT_SPECIFIC_HEAD=1
    export RUMPL_TRAIN_SCOPE=joint_specific_head
    ;;
  graph_residual)
    export CODE_OVERRIDE="R7c_${lr_variant}_seed${seed}"
    export TAG_OVERRIDE="R7c_H76_outputSide_postPFTGraphResidual_w322_${lr_variant}_seed${seed}_20260809"
    export CONTROL_NOTE_OVERRIDE="H76 frozen deterministic; zero-init post-PFT skeleton graph residual; lr=${lr}; 5 epochs"
    export RUMPL_POST_PFT_GRAPH_RESIDUAL=1
    export RUMPL_TRAIN_SCOPE=post_pft_graph_residual
    ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac

export RUMPL_TRI_ANCHOR=1
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_FINETUNE_LR="${lr}"
export RUMPL_END_EPOCH=5
export RUMPL_DISABLE_FIXED_VIEW_CURRICULUM=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export RUMPL_WORKERS="${RUMPL_WORKERS:-12}"

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
