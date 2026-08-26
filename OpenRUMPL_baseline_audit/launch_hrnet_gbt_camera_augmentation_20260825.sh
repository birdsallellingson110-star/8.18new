#!/usr/bin/env bash
# Conservative continuation of the clean-matched canonical HRNet generator.
# This branch is isolated and is retained only if clean accuracy remains
# matched and unseen-camera audits improve.
set -euo pipefail

ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/best_available_modules_20260825_safe_candidates/hrnet

export REPAIR_NAME="${CAM_AUG_NAME:-gbt_camera_aug6_lr1e6_drop10_synth50}"
export REPAIR_LR="${CAM_AUG_LR:-1e-6}"
export REPAIR_EPOCHS="${CAM_AUG_EPOCHS:-6}"
export REPAIR_LR_STEPS="${CAM_AUG_LR_STEPS:-4}"
export REPAIR_PELVIS_PRIOR=1
export REPAIR_BODY_REG=1e-2
export REPAIR_VISIBLE_GPU=1
export REPAIR_INIT
REPAIR_INIT=$(cat "${ROOT}/generator/checkpoint.txt")

# GBT-style robustness controls. One of four real views is replaced on half
# the samples, so an internal V2 subset can never be entirely synthetic.
export REPAIR_GBT_TOKEN_DROPOUT="${CAM_AUG_TOKEN_DROPOUT:-0.10}"
export REPAIR_GBT_TOKEN_DROPOUT_EPOCHS="${CAM_AUG_TOKEN_DROPOUT_EPOCHS:-${REPAIR_EPOCHS}}"
export REPAIR_SYNTHETIC_REPLACE_PROB="${CAM_AUG_SYNTHETIC_PROB:-0.50}"
export REPAIR_SYNTHETIC_RADIUS_MIN_M=3.0
export REPAIR_SYNTHETIC_RADIUS_MAX_M=6.0
export REPAIR_SYNTHETIC_HEIGHT_MIN_M=1.0
export REPAIR_SYNTHETIC_HEIGHT_MAX_M=2.5

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_hrnet_canonical_repair_branch_20260825.sh
