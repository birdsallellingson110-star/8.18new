#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
cd "$repo"

export RUMPL_GLOBAL_JOINT_VIEW_FUSION=1
export RUMPL_GLOBAL_JOINT_VIEW_DEPTH=2
export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=0
export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=0
export RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT=0.05
export RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE=1
export RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT=0.12
export GBT_LEARNABLE_BIAS=0
export GBT_LEARNED_RELIABILITY=0

exec "$repo/run_official_like_cmu_seed0_20260722.sh" \
  0 J2_adaptive_globalJV_d2_nobias_g005_012_seed0_20260723 0 1 16
