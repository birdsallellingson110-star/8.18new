#!/usr/bin/env bash
# H81-H83: independent, identity-initialized responses to H76 per-joint residual conflicts.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {joint_gate|graph_residual|joint_head} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H81_H83_targeted_pft
export SEED_OVERRIDE=0
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_ANCHOR_CENTER_PER_JOINT=0
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0

case "${variant}" in
  joint_gate)
    export CODE_OVERRIDE=H81
    export TAG_OVERRIDE=H81_H76_perJointResidualGate_workers12_seed0_20260803
    export CONTROL_NOTE_OVERRIDE="H76 plus 17 learnable per-joint residual scales initialized to one"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    ;;
  graph_residual)
    export CODE_OVERRIDE=H82
    export TAG_OVERRIDE=H82_H76_postPFTGraphResidual_workers12_seed0_20260803
    export CONTROL_NOTE_OVERRIDE="H76 plus zero-initialized H36M skeleton message after PFT"
    export RUMPL_POST_PFT_GRAPH_RESIDUAL=1
    ;;
  joint_head)
    export CODE_OVERRIDE=H83
    export TAG_OVERRIDE=H83_H76_jointSpecificHead_workers12_seed0_20260803
    export CONTROL_NOTE_OVERRIDE="H76 plus zero-initialized joint-specific correction beside shared head"
    export RUMPL_JOINT_SPECIFIC_HEAD=1
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
