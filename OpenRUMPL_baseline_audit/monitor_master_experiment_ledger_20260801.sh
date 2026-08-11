#!/usr/bin/env bash
# Refresh the mounted-disk experiment ledger while the controlled queues live.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
COLLECTOR=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/collect_master_experiment_ledger_20260801.py
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOCK=${ROOT}/master_experiment_ledger_monitor.lock

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "[ledger] another monitor already owns ${LOCK}"
  exit 0
fi

echo "[ledger] monitor start $(date --iso-8601=seconds)"
while pgrep -f 'H34_a1d_nobias_triAnchor|H37_J[01]_globalJV1|H39_U0_undistortPoints|queue_H3[5-8]_' >/dev/null; do
  "${PY}" "${COLLECTOR}"
  sleep 60
done
"${PY}" "${COLLECTOR}"
echo "[ledger] monitor end $(date --iso-8601=seconds)"
