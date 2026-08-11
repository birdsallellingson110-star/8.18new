#!/usr/bin/env bash
# Run after H135 completes (or in parallel on another GPU if free).
set -euo pipefail
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
# shellcheck source=/dev/null
source "${AUDIT}/experiment_should_skip.sh"
LAUNCH=${AUDIT}/launch_H135_H140_beat_gbt_phase2_20260805.sh
LOG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/queue_beat_gbt_phase2_remainder.log
GPU=${1:-0}
exec >>"${LOG}" 2>&1
echo "remainder start $(date --iso-8601=seconds) GPU=${GPU}"
for v in h136_temporal_h81_unfreeze h137_h81_depro_caa h138_h114_v4w h139_h117_temporal_h76 h140_h81_gbt_bias_ft; do
  echo "=== ${v} ==="
  if experiment_should_skip_variant "${v}" 2>/dev/null; then
    echo "[remainder] skip ${v} (skip registry)"
    continue
  fi
  bash "${LAUNCH}" "${v}" "${GPU}"
done
echo "remainder done $(date --iso-8601=seconds)"
