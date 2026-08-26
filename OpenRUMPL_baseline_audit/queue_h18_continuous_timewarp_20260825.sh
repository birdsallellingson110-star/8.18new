#!/usr/bin/env bash
set -euo pipefail

CURRENT_PID=388559
CURRENT_ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/best_available_modules_20260825_safe_candidates/hrnet/canonical_h18/model_batch8_accum8_eval32

while kill -0 "${CURRENT_PID}" 2>/dev/null; do
  sleep 30
done
test -f "${CURRENT_ROOT}/COMPLETED"
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_h18_continuous_timewarp_20260825.sh
