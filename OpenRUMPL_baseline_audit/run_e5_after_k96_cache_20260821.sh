#!/usr/bin/env bash
set -euo pipefail
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e5_ray_dual_stream
mkdir -p "${OUT}"

"${AUDIT}/launch_build_k96_temporal_caches_20260821.sh"

CUDA_VISIBLE_DEVICES=0 "${AUDIT}/launch_e5_ray_conditioned_dual_stream_20260821.sh" \
  ray 0 >"${OUT}/ray_seed0.log" 2>&1 &
ray_pid=$!
CUDA_VISIBLE_DEVICES=0 "${AUDIT}/launch_e5_ray_conditioned_dual_stream_20260821.sh" \
  control 0 >"${OUT}/control_seed0.log" 2>&1 &
control_pid=$!
wait "${ray_pid}"
wait "${control_pid}"
