#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT}
tag=${2:?}
checkpoint=${3:?}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit

export RUMPL_GATED_JOINT_ADAPTER=1
export RUMPL_JOINT_ADAPTER_INDICES=2,5,8
export RUMPL_JOINT_ADAPTER_SCALE_INIT=1.2
export RUMPL_ALT_JOINT_VIEW=0 RUMPL_VFT_DEPTH=12 RUMPL_PFT_DEPTH=12
export RUMPL_MULTI_HYP=1 RUMPL_FIX_PFT_LAST_BLOCK=0
for n_views in 2 3 4 5; do
  bash "$repo/eval_exact_multiview_20260723.sh" \
    "$gpu" "${tag}_mmpose_v${n_views}" "$checkpoint" "$n_views" baseline
done
