#!/usr/bin/env bash
# Confirm H63/H69 across RUMPL seeds and test their non-conflicting fusion.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {h63_seed1|h63_seed2|h69_seed1|h69_seed2|fusion_seed0} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
export BASE_OVERRIDE=${ROOT}/H71_H75_confirm_fuse
export RUMPL_ANCHOR_CENTERED_RAYS=0
export RUMPL_INPUT_PLUCKER=0
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0

case "${variant}" in
  fusion_seed0)
    export CODE_OVERRIDE=H71
    export TAG_OVERRIDE=H71_H63input_anchorCenteredRays_workers12_seed0_20260803
    export SEED_OVERRIDE=0
    export H21_OVERRIDE=${ROOT}/H55_H58_h21_screen/H62_balanced10k_seed0/final.pth
    export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_h62_balanced10k_legswap
    export RUMPL_ANCHOR_CENTERED_RAYS=1
    export CONTROL_NOTE_OVERRIDE="H63 balanced10k input plus independently successful H69 anchor-centered rays"
    ;;
  h63_seed1|h63_seed2)
    seed=${variant##*seed}
    code=$((71 + seed))
    export CODE_OVERRIDE=H${code}
    export TAG_OVERRIDE=H${code}_H63balanced10kInput_workers12_seed${seed}_20260803
    export SEED_OVERRIDE=${seed}
    export H21_OVERRIDE=${ROOT}/H55_H58_h21_screen/H62_balanced10k_seed0/final.pth
    export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_h62_balanced10k_legswap
    export CONTROL_NOTE_OVERRIDE="H63 paired RUMPL replication seed=${seed}"
    ;;
  h69_seed1|h69_seed2)
    seed=${variant##*seed}
    code=$((73 + seed))
    export CODE_OVERRIDE=H${code}
    export TAG_OVERRIDE=H${code}_H69anchorCenteredRays_workers12_seed${seed}_20260803
    export SEED_OVERRIDE=${seed}
    export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
    export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
    export RUMPL_ANCHOR_CENTERED_RAYS=1
    export CONTROL_NOTE_OVERRIDE="H69 paired RUMPL replication seed=${seed}"
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${physical_gpu}"
