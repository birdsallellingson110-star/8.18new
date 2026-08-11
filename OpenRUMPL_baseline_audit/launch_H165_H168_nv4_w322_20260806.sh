#!/usr/bin/env bash
# H165+: minimal-fusion hypotheses with fixed 4-view capacity (post root-cause fix).
set -euo pipefail

physical_gpu=${1:-0}
variant=${2:-h165_tri_only}

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_TRI_ANCHOR=1
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export CODE_OVERRIDE=H165
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_EVAL_STRICT=0
export RUMPL_INIT_CHECKPOINT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H81_H76_perJointResidualGate_workers12_seed0_20260803_2026-08-03_12-55-56/model_best.pth.tar
export RUMPL_STACK_FROM=H81
export RUMPL_FINETUNE_LR=5e-5
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export BASE_OVERRIDE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H165_H168_nv4_w322
export RUMPL_WORKERS=12

case "${variant}" in
  h165_tri_only)
    export TAG_OVERRIDE=H165_H81_triAnchorOnly_skipVftPft_w322_nv4_workers12_seed0_20260806
    export RUMPL_SKIP_VFT=1
    export RUMPL_SKIP_PFT=1
    export RUMPL_TRI_ANCHOR=1
    ;;
  h166_vft1_tri)
    export CODE_OVERRIDE=H166
    export TAG_OVERRIDE=H166_H81_vftDepth1_triAnchor_w322_nv4_workers12_seed0_20260806
    export RUMPL_VFT_DEPTH=1
    export RUMPL_SKIP_PFT=0
    export RUMPL_TRI_ANCHOR=1
    ;;
  h167_shallow_no_tri)
    export CODE_OVERRIDE=H167
    export TAG_OVERRIDE=H167_H81_vftDepth1_noTri_w322_nv4_workers12_seed0_20260806
    export RUMPL_VFT_DEPTH=1
    export RUMPL_TRI_ANCHOR=0
    export RUMPL_TRI_ANCHOR_REG=0
    ;;
  h168_gate_baseline)
    export CODE_OVERRIDE=H168
    export TAG_OVERRIDE=H168_H81_perJointGate_baseline_w322_nv4_workers12_seed0_20260806
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    unset RUMPL_SKIP_VFT RUMPL_SKIP_PFT RUMPL_VFT_DEPTH
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 1
    ;;
esac

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
