#!/usr/bin/env bash
# Clean official evaluation against R5: mmpose V2--V5 only.
# usage: $0 GPU MODE TAG CHECKPOINT
set -euo pipefail

gpu=${1:?usage: $0 GPU MODE TAG CHECKPOINT}
mode=${2:?}
tag=${3:?}
checkpoint=${4:?}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit

export RUMPL_MULTI_HYP=1
export RUMPL_FIX_PFT_LAST_BLOCK=1
case "$mode" in
  t0_fix)
    export RUMPL_ALT_JOINT_VIEW=0 RUMPL_VFT_DEPTH=12 RUMPL_PFT_DEPTH=12
    ;;
  t1_shallow)
    export RUMPL_ALT_JOINT_VIEW=0 RUMPL_VFT_DEPTH=4 RUMPL_PFT_DEPTH=8
    ;;
  t2_alt)
    export RUMPL_ALT_JOINT_VIEW=1 RUMPL_ALT_JOINT_VIEW_DEPTH=4
    export RUMPL_VFT_DEPTH=0 RUMPL_PFT_DEPTH=4
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

for n_views in 2 3 4 5; do
  bash "$repo/eval_exact_multiview_20260723.sh" \
    "$gpu" "${tag}_mmpose_v${n_views}" "$checkpoint" "$n_views" baseline
done
