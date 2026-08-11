#!/usr/bin/env bash
# GF-M0/M1: CVPR'22 GraFormer core replaces RUMPL PFT after unchanged VFT.
# Usage: bash launch_GF_M0_M1_graformer_pft_20260809.sh {attention|full} GPU
set -euo pipefail

mode=${1:?expected attention or full}
physical_gpu=${2:-0}
case "${mode}" in
  attention)
    code=GF-M0
    tag=GF_M0_modelAT_GraAttentionPFT_depth5_seed0_20260809
    note="GraFormer paper model-AT ablation: input/output ChebGConv plus 5 GraAttention layers replace RUMPL PFT after unchanged VFT"
    ;;
  full)
    code=GF-M1
    tag=GF_M1_fullGraFormerPFT_depth5_seed0_20260809
    note="GraFormer CVPR22 full model: input/output ChebGConv plus 5 alternating GraAttention and residual ChebGConv blocks replace RUMPL PFT after unchanged VFT"
    ;;
  *)
    echo "unknown GraFormer PFT mode: ${mode}" >&2
    exit 2
    ;;
esac

ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

# Model-only comparison.  Exclude distillation, temporal paths, auxiliary
# objectives, pretrained parents, and every previous RUMPL adapter ablation.
unset RUMPL_INIT_CHECKPOINT RUMPL_STACK_FROM RUMPL_LOSS_TYPE || true
unset STUDENT_GT_W DISTILL_LAMBDA HARD_VIEW_MINING STUDENT_VIEWS || true
unset DISTILL_TEACHER_EVAL LEG_DISTILL_W TEMPORAL_LAMBDA || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0
export MONO_W=0 MONO_GT_W=0
export RUMPL_TRAIN_SCOPE=all

export CODE_OVERRIDE="${code}"
export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/GraFormer_PFT_20260809
export TAG_OVERRIDE="${tag}"
export SEED_OVERRIDE=0
export CONTROL_NOTE_OVERRIDE="${note}; official defaults N=5 heads=4 dropout=0.25; H76 training/data/loss/sampling unchanged"

# Retain the H76 ray representation, view fusion, 3-D head and anchor.
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_SEMANTIC_GRAPH_PRE_VFT=off
export RUMPL_GRAFORMER_PFT="${mode}"
export RUMPL_GRAFORMER_DEPTH=5

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
