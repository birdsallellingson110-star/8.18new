#!/usr/bin/env bash
# H70: condition retained RUMPL on analytic ray-intersection uncertainty.
set -euo pipefail

physical_gpu=${1:-1}
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export CODE_OVERRIDE=H70
export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H70_h50_geometry_uncertainty
export TAG_OVERRIDE=H70_H50_geometryUncertaintyToken_workers12_seed0_20260803
export CONTROL_NOTE_OVERRIDE="zero-init analytic ray-normal-matrix uncertainty embedding on the RUMPL fusion token"
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=1
export RUMPL_INPUT_PLUCKER=0
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_ANCHOR_CENTERED_RAYS=0

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
