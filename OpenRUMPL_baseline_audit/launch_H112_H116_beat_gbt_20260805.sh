#!/usr/bin/env bash
# H112-H116: single-frame RUMPL variants on H76 input stack to close V4 gap vs GBT-HRNet.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {h112_vft_bias|h113_reproj|h114_v4_train_weight|h115_global_jv|h116_h81_reproj} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H112_H116_beat_gbt
export SEED_OVERRIDE=0
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_ANCHOR_CENTER_PER_JOINT=0
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_GBT_SET_DECODER=0
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1

case "${variant}" in
  h112_vft_bias)
    export CODE_OVERRIDE=H112
    export TAG_OVERRIDE=H112_H76_VFT_gbtConfGeomBias_ftH76_workers12_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + GBT conf+geom bias in VFT"
    export GBT_LEARNABLE_BIAS=1
    export GBT_USE_CONF_BIAS=1
    export GBT_USE_GEOM_BIAS=1
    ;;
  h113_reproj)
    export CODE_OVERRIDE=H113
    export TAG_OVERRIDE=H113_H76_reprojLambda001_ftH76_workers12_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + reprojection loss lambda=0.01"
    export REPROJ_LAMBDA=0.01
    ;;
  h114_v4_train_weight)
    export CODE_OVERRIDE=H114
    export TAG_OVERRIDE=H114_H76_viewWeights124_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H76 with stronger 4-view sampling weight 1:2:4 for V2:V3:V4"
    export RUMPL_VIEW_COUNT_WEIGHTS=1,2,4
    ;;
  h115_global_jv)
    export CODE_OVERRIDE=H115
    export TAG_OVERRIDE=H115_H76_globalJV1_rezero_ftH76_workers12_seed0_20260805
    export RUMPL_STACK_FROM=H76
    export CONTROL_NOTE_OVERRIDE="H76 ckpt + ReZero global JV depth=1"
    export GBT_GLOBAL_JV_DEPTH=1
    export GBT_GLOBAL_JV_GATED=1
    export GBT_GLOBAL_JV_BIASED=0
    ;;
  h116_h81_reproj)
    export CODE_OVERRIDE=H116
    export TAG_OVERRIDE=H116_H81_perJointGate_reproj001_ftH81_workers12_seed0_20260805
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 ckpt + reprojection lambda=0.01"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export REPROJ_LAMBDA=0.01
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
