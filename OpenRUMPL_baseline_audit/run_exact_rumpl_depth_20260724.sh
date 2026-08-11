#!/usr/bin/env bash
# Train RUMPL with triangulation-anchored residual head and/or per-view ray-depth aux supervision.
# usage: $0 GPU VARIANT TRI_ANCHOR RAY_DEPTH_AUX [AUX_WEIGHT]
set -euo pipefail

gpu=${1:?usage: $0 GPU VARIANT TRI_ANCHOR RAY_DEPTH_AUX [AUX_WEIGHT]}
variant=${2:?}
tri_anchor=${3:?}
ray_depth_aux=${4:?}
aux_weight=${5:-0.1}

for v in "$tri_anchor" "$ray_depth_aux"; do
  if [[ "$v" != "0" && "$v" != "1" ]]; then
    echo "TRI_ANCHOR/RAY_DEPTH_AUX must be 0 or 1" >&2
    exit 2
  fi
done

export RUMPL_TRI_ANCHOR="$tri_anchor"
export RUMPL_RAY_DEPTH_AUX="$ray_depth_aux"
export RUMPL_RAY_DEPTH_AUX_WEIGHT="$aux_weight"

echo "RUMPL_TRI_ANCHOR=$RUMPL_TRI_ANCHOR RUMPL_RAY_DEPTH_AUX=$RUMPL_RAY_DEPTH_AUX RUMPL_RAY_DEPTH_AUX_WEIGHT=$RUMPL_RAY_DEPTH_AUX_WEIGHT"

# Same protocol as R5 baseline: public PFT behavior, corrected scheduler, 16 workers.
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
