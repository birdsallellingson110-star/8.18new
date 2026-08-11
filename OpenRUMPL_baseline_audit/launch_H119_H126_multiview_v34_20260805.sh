#!/usr/bin/env bash
# H119-H126: multi-camera-aware training/fusion for V3/V4 (HRNet→A1D→H21, H76/H81 stack).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 VARIANT PHYSICAL_GPU" >&2
  echo "Variants: h120_w322 h121_h81_w322 h122_relview h123_gjv2 h124_mono" >&2
  echo "          h127_mono_h81 h128_relview_h81 (deprecated: h119/h125/h126 always-N-view)" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H119_H126_multiview_v34
export SEED_OVERRIDE=0
export RUMPL_WORKERS=${RUMPL_WORKERS:-8}

export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_ANCHOR_CENTER_PER_JOINT=0
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_GBT_SET_DECODER=0
export GBT_LEARNABLE_BIAS=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_GATED=0
export GBT_GLOBAL_JV_BIASED=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0
export MONO_W=0
export MONO_GT_W=0
export MONO_MARGIN=0

# Paper gate: keep 2-view training mass; never train-only-3/4-view (hurts V2 at test).

# Default curriculum matches H76/H81: 2 views epochs 0–7, then random 2/3/4.
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_RANDOM_VIEW_SUBSET=1
# 3:2:2 ≈ 50% / 33% / 17% for V2/V3/V4 counts — boosts 3/4 without abandoning 2-view.
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2

case "${variant}" in
  h119_always4|h125_h81_always4|h126_always3)
    echo "DEPRECATED: ${variant} sacrifices V2; use h127/h128/h122 instead" >&2
    exit 2
    ;;
  h120_w322)
    export CODE_OVERRIDE=H120
    export TAG_OVERRIDE=H120_H76_viewWeights322_ftH76_workers8_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + 3:2:2 view-count sampling"
    ;;
  h121_h81_w322)
    export CODE_OVERRIDE=H121
    export TAG_OVERRIDE=H121_H81_perJointGate_viewWeights322_ftH81_workers8_seed0_20260805
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 ckpt + 3:2:2 sampling"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    ;;
  h122_relview)
    echo "DEPRECATED: h122 relview-only superseded by H130 (relview+mask); see EXPERIMENT_DEDUP_REGISTRY" >&2
    exit 2
    ;;
  h123_gjv2)
    export CODE_OVERRIDE=H123
    export TAG_OVERRIDE=H123_H76_globalJV2_rezero_w322_ftH76_workers8_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + ReZero global JV depth=2"
    export GBT_GLOBAL_JV_DEPTH=2
    export GBT_GLOBAL_JV_GATED=1
    ;;
  h124_mono)
    export CODE_OVERRIDE=H124
    export TAG_OVERRIDE=H124_H76_nestedViewMono_w005_w322_ftH76_workers8_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + nested-view mono loss"
    export MONO_W=0.05
    export MONO_GT_W=0.0
    export MONO_MARGIN=0.001
    ;;
  h127_mono_h81)
    export CODE_OVERRIDE=H127
    export TAG_OVERRIDE=H127_H81_perJointGate_mono005_w322_ftH81_workers8_seed0_20260805
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 ckpt + mono loss + 3:2:2"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export MONO_W=0.05
    export MONO_GT_W=0.0
    export MONO_MARGIN=0.001
    ;;
  h128_relview_h81)
    export CODE_OVERRIDE=H128
    export TAG_OVERRIDE=H128_H81_relViewFusion_w322_ftH81_workers8_seed0_20260805
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 ckpt + relative view fusion + 3:2:2"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_RELATIVE_VIEW_FUSION=1
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

for _base in "${ROOT}/H112_H116_beat_gbt" "${ROOT}/H119_H126_multiview_v34" "${ROOT}/H129_H134_literature_fusion"; do
  if [[ -s "${_base}/completed/${TAG_OVERRIDE}.done" ]]; then
    echo "[${CODE_OVERRIDE}] skip completed ${TAG_OVERRIDE}"
    exit 0
  fi
done
unset _base

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
