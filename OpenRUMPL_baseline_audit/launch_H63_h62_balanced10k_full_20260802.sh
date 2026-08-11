#!/usr/bin/env bash
# H63: reuse the audited H59 full chain for the H62 10k-step winner.
set -euo pipefail

export CODE_OVERRIDE=H63
export H21_OVERRIDE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H55_H58_h21_screen/H62_balanced10k_seed0/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_h62_balanced10k_legswap
export BASE_OVERRIDE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H63_h62_balanced10k_full
export TAG_OVERRIDE=H63_H62balanced10kH21_RUMPL_workers12_seed0_20260802
export CONTROL_NOTE_OVERRIDE="H58 balanced H21 continued from 5000 to 10000 steps"

exec bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H59_h58_balanced_full_20260802.sh "${1:-1}"
