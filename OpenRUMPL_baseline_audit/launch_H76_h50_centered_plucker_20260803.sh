#!/usr/bin/env bash
# H76: combine H67's weak-pair Plucker benefit with H69's centered coordinates.
set -euo pipefail

physical_gpu=${1:-1}
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export CODE_OVERRIDE=H76
export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H76_h50_centered_plucker
export TAG_OVERRIDE=H76_H50_anchorCenteredPlucker_workers12_seed0_20260803
export SEED_OVERRIDE=0
export CONTROL_NOTE_OVERRIDE="anchor-centered Plucker rays; targeted combination of H67 and H69 representations"
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
