#!/usr/bin/env bash
# SG-M0/M1: AAAI'24 SGraFormer spatial encoder before the original RUMPL VFT.
# Usage: bash launch_SG_M0_M1_semantic_graph_20260809.sh {position|full} GPU
set -euo pipefail

mode=${1:?expected position or full}
physical_gpu=${2:-0}
case "${mode}" in
  position)
    code=SG-M0
    tag=SG_M0_positionOnly_preVFT_depth4_seed0_20260809
    note="SGraFormer paper ablation: position-only per-view spatial Transformer before unchanged RUMPL VFT"
    ;;
  full)
    code=SG-M1
    tag=SG_M1_fullSemanticGraph_preVFT_depth4_seed0_20260809
    note="SGraFormer AAAI24 full semantic graph encoder (position+4-hop spatial+edge) before unchanged RUMPL VFT"
    ;;
  *)
    echo "unknown semantic graph mode: ${mode}" >&2
    exit 2
    ;;
esac

ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

# This is a model-only comparison.  Explicitly exclude every distillation,
# auxiliary-loss, adapter-only, temporal, or parent-checkpoint setting.
unset RUMPL_INIT_CHECKPOINT RUMPL_STACK_FROM RUMPL_LOSS_TYPE || true
unset STUDENT_GT_W DISTILL_LAMBDA HARD_VIEW_MINING STUDENT_VIEWS || true
unset DISTILL_TEACHER_EVAL LEG_DISTILL_W TEMPORAL_LAMBDA || true
export DISTILL_W=0
export FEAT_DISTILL_W=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0
export MONO_W=0
export MONO_GT_W=0
export RUMPL_TRAIN_SCOPE=all

export CODE_OVERRIDE="${code}"
export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/SGraFormer_preVFT_20260809
export TAG_OVERRIDE="${tag}"
export SEED_OVERRIDE=0
export CONTROL_NOTE_OVERRIDE="${note}; training/data/loss/sampling identical to H76"

# H76 representation and output path are retained exactly.
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_SEMANTIC_GRAPH_PRE_VFT="${mode}"
export RUMPL_SEMANTIC_GRAPH_DEPTH=4

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"

