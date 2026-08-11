#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT INDICES VIEW_POWER [SCALE_INIT]}
tag=${2:?}
checkpoint=${3:?}
indices=${4:?}
view_power=${5:?}
scale_init=${6:-1.2}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit

export RUMPL_GATED_JOINT_ADAPTER=1
export RUMPL_JOINT_ADAPTER_INDICES="$indices"
export RUMPL_JOINT_ADAPTER_SCALE_INIT="$scale_init"
export RUMPL_JOINT_ADAPTER_VIEW_POWER="$view_power"
export RUMPL_ALT_JOINT_VIEW=0 RUMPL_VFT_DEPTH=12 RUMPL_PFT_DEPTH=12
export RUMPL_MULTI_HYP=1 RUMPL_FIX_PFT_LAST_BLOCK=0
for n_views in 2 3 4 5; do
  bash "$repo/eval_exact_multiview_20260723.sh" \
    "$gpu" "${tag}_mmpose_v${n_views}" "$checkpoint" "$n_views" baseline
done
