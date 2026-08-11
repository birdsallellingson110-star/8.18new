#!/usr/bin/env bash
# After H135 finishes, run H136-H140 sequentially on GPU0.
set -euo pipefail
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
LAUNCH=${AUDIT}/launch_H135_H140_beat_gbt_phase2_20260805.sh
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
DONE=${ROOT}/H135_H140_beat_gbt_phase2/completed_H135_temporal_h81_T9_biased_frozen.done
LOG=${ROOT}/H135_H140_beat_gbt_phase2/logs/chain_h136_h140.log

exec >>"${LOG}" 2>&1
echo "[chain] wait H135 $(date --iso-8601=seconds)"
while [[ ! -s "${DONE}" ]]; do sleep 120; done
echo "[chain] H135 done, run H136-H140"
for v in h136_temporal_h81_unfreeze h137_h81_depro_caa h138_h114_v4w h139_h117_temporal_h76 h140_h81_gbt_bias_ft; do
  echo "[chain] ${v} $(date --iso-8601=seconds)"
  bash "${LAUNCH}" "${v}" 0
done
echo "[chain] finished $(date --iso-8601=seconds)"
