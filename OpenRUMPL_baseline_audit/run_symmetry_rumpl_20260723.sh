#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
cd "$repo"

export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export GBT_LEARNABLE_BIAS=0
export GBT_LEARNED_RELIABILITY=0
export RUMPL_SYMMETRY_LOSS_WEIGHT=0.5

exec "$repo/run_official_like_cmu_seed0_20260722.sh" \
  1 S1_rumpl_symmetry_w05_seed0_20260723 0 1 16
