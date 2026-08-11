#!/usr/bin/env bash
# R2a: H76 frozen + explicit mean 2D confidence bias at VFT fusion rows.
set -euo pipefail

variant=${1:?usage: $0 <lr1e4|lr5e4> <physical_gpu>}
physical_gpu=${2:?usage: $0 <lr1e4|lr5e4> <physical_gpu>}
case "${variant}" in
  lr1e4) lr=1e-4 ;;
  lr5e4) lr=5e-4 ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
export CODE_OVERRIDE="R2a_${variant}"
export BASE_OVERRIDE="${ROOT}/R2a_confidence_view_bias_20260808"
export TAG_OVERRIDE="R2a_H76_confidenceViewBias_w322_${variant}_seed0_20260808"
export CONTROL_NOTE_OVERRIDE="H76 frozen; explicit mean-joint 2D confidence fusion-row bias; 5 epochs; lr=${lr}"
export SEED_OVERRIDE=0
export RUMPL_STACK_FROM=H76
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export H21_OVERRIDE="${ROOT}/H55_H58_h21_screen/H58_balanced_views_seed0/final.pth"
export RUMPL_CONFIDENCE_VIEW_BIAS=1
export RUMPL_TRAIN_SCOPE=confidence_view_bias
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_FINETUNE_LR="${lr}"
export RUMPL_END_EPOCH=5
export RUMPL_DISABLE_FIXED_VIEW_CURRICULUM=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export RUMPL_WORKERS="${RUMPL_WORKERS:-12}"

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
