#!/usr/bin/env bash
# Strict H50 ablations: change only the ray representation seen by retained RUMPL.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {plucker|harmonic15|anchor_centered} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H67_H69_h50_representation
export RUMPL_INPUT_PLUCKER=0
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_ANCHOR_CENTERED_RAYS=0

case "${variant}" in
  plucker)
    export CODE_OVERRIDE=H67
    export TAG_OVERRIDE=H67_H50_plucker_workers12_seed0_20260803
    export CONTROL_NOTE_OVERRIDE="Plucker line representation; all H50 settings fixed"
    export RUMPL_INPUT_PLUCKER=1
    ;;
  harmonic15)
    export CODE_OVERRIDE=H68
    export TAG_OVERRIDE=H68_H50_harmonic15_workers12_seed0_20260803
    export CONTROL_NOTE_OVERRIDE="harmonic ray encoding L=15; all H50 settings fixed"
    export RUMPL_INPUT_HARMONIC_L=15
    ;;
  anchor_centered)
    export CODE_OVERRIDE=H69
    export TAG_OVERRIDE=H69_H50_anchorCenteredRays_workers12_seed0_20260803
    export CONTROL_NOTE_OVERRIDE="triangulation-anchor-centered rays; all H50 settings fixed"
    export RUMPL_ANCHOR_CENTERED_RAYS=1
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
