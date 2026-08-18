#!/usr/bin/env bash
# E3: H76 + parallel GBT/MVGFormer-style joint-query 3D residual.
# The original H76 path is loaded and frozen; only the zero-initialized
# query decoder is trained.  Variant {global|local} controls whether a joint
# query can inspect all joint/view tokens or only its own view set.
set -euo pipefail

variant=${1:?usage: $0 <global|local> <physical_gpu> [seed]}
physical_gpu=${2:?usage: $0 <global|local> <physical_gpu> [seed]}
seed=${3:-0}
case "${variant}" in
  global) global=1; desc=globalJointMemory ;;
  local)  global=0; desc=perJointMemory ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
export CODE_OVERRIDE="E3b_${variant}"
export BASE_OVERRIDE="${ROOT}/GBT_Query_Residual_E3b_20260814"
export TAG_OVERRIDE="E3b_H76_GBTQueryResidual_${desc}_w322_seed${seed}_20260814"
export CONTROL_NOTE_OVERRIDE="H76 frozen + zero-init GBT/MVGFormer joint-query 3D residual; ${desc}; depth=2"
export SEED_OVERRIDE="${seed}"
export RUMPL_STACK_FROM=H76
export RUMPL_TRAIN_SCOPE=gbt_query_residual
export RUMPL_FINETUNE_LR=1e-4
export RUMPL_END_EPOCH=20
export RUMPL_DISABLE_FIXED_VIEW_CURRICULUM=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
export RUMPL_WORKERS="${RUMPL_WORKERS:-12}"
export RUMPL_GBT_QUERY_RESIDUAL=1
export RUMPL_GBT_QUERY_RESIDUAL_GLOBAL="${global}"
export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5

export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export H21_OVERRIDE="${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth"
export RUMPL_TRI_ANCHOR=1
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_ANCHOR_CENTER_PER_JOINT=0
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_PFT_REPEAT_LAST=1

# Explicitly isolate E3 from all previously tested branches.
export RUMPL_GBT_SET_DECODER=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0
export RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0
export DEPRO_LAMBDA=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0
export MONO_W=0
export MONO_GT_W=0
export MONO_MARGIN=0

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
