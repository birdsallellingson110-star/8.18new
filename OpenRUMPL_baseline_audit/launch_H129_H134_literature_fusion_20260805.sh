#!/usr/bin/env bash
# H129-H134: literature-backed fusion (Gifformer mask, MTF relview, ESM reproj+mono, GBT biased GJV).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 VARIANT PHYSICAL_GPU" >&2
  echo "Variants: h129_mask05 h130_relview_mask05 h131_h81_reproj_mono h132_gjv1_biased h133_w322_only h134_relview_mono" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H129_H134_literature_fusion
export SEED_OVERRIDE=0
export RUMPL_WORKERS=${RUMPL_WORKERS:-6}

export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_GATED=0
export GBT_GLOBAL_JV_BIASED=0
export VFT_FULL_RANDOM_MASK=0
export REPROJ_LAMBDA=0
export MONO_W=0
export MONO_GT_W=0
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2

case "${variant}" in
  h129_mask05)
    echo "DEPRECATED: same mechanism as H43-H45 on H35; see EXPERIMENT_DEDUP_REGISTRY" >&2
    exit 2
    ;;
  h130_relview_mask05)
    export CODE_OVERRIDE=H130
    export TAG_OVERRIDE=H130_H76_relViewFusion_gifformerMask05_w322_ftH76_workers6_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + RelativeViewFusion + VFT mask M=0.5"
    export RUMPL_RELATIVE_VIEW_FUSION=1
    export VFT_FULL_RANDOM_MASK=0.5
    ;;
  h131_h81_reproj_mono)
    export CODE_OVERRIDE=H131
    export TAG_OVERRIDE=H131_H81_reproj001_mono005_w322_ftH81_workers6_seed0_20260805
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 ckpt + reproj+mono; 3:2:2 sampling"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export REPROJ_LAMBDA=0.01
    export MONO_W=0.05
    export MONO_MARGIN=0.001
    ;;
  h132_gjv1_biased)
    export CODE_OVERRIDE=H132
    export TAG_OVERRIDE=H132_H76_globalJV1_gbtBiased_rezero_w322_ftH76_workers6_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + GBT-biased global JV depth=1 ReZero"
    export GBT_GLOBAL_JV_DEPTH=1
    export GBT_GLOBAL_JV_GATED=1
    export GBT_GLOBAL_JV_BIASED=1
    ;;
  h133_w322_only)
    echo "DEPRECATED: duplicates H120/H76 w322 curriculum probe; cancelled in dedup registry" >&2
    exit 2
    ;;
  h134_relview_mono)
    echo "DEPRECATED: redundant with H130+H124; cancelled in dedup registry" >&2
    exit 2
    ;;
  *)
    echo "unsupported: ${variant}" >&2
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
