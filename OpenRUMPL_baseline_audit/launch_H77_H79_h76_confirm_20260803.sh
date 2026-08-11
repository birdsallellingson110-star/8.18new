#!/usr/bin/env bash
# Three controlled confirmations of the H76 centered-Plucker winner.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {seed1|seed2|replay0} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H77_H79_h76_confirm
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0

case "${variant}" in
  seed1)
    export CODE_OVERRIDE=H77
    export TAG_OVERRIDE=H77_H76anchorCenteredPlucker_workers12_seed1_20260803
    export SEED_OVERRIDE=1
    ;;
  seed2)
    export CODE_OVERRIDE=H78
    export TAG_OVERRIDE=H78_H76anchorCenteredPlucker_workers12_seed2_20260803
    export SEED_OVERRIDE=2
    ;;
  replay0)
    export CODE_OVERRIDE=H79
    export TAG_OVERRIDE=H79_H76anchorCenteredPlucker_exactReplay_workers12_seed0_20260803
    export SEED_OVERRIDE=0
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac
export CONTROL_NOTE_OVERRIDE="H76 centered-Plucker confirmation ${variant}"

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
