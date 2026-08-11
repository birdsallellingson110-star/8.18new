#!/usr/bin/env bash
# Refresh the reproducibility ledger until H35 and all H41-H45 chains finish.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
COLLECTOR=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/collect_master_experiment_ledger_20260801.py
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/H41_H45_paper_mask_fusion/ledger_monitor.log

mkdir -p "$(dirname "${LOG}")"
while pgrep -f 'run/train_rumpl.py.*(H35_a1dH21|H4[1-5]_)' >/dev/null \
  || pgrep -f 'launch_H41_H45_paper_mask_fusion_20260801.sh' >/dev/null; do
  echo "[ledger] refresh $(date --iso-8601=seconds)" >>"${LOG}"
  "${PY}" "${COLLECTOR}" >>"${LOG}" 2>&1
  sleep 300
done

echo "[ledger] final refresh $(date --iso-8601=seconds)" >>"${LOG}"
"${PY}" "${COLLECTOR}" >>"${LOG}" 2>&1
touch "${ROOT}/H41_H45_paper_mask_fusion/ledger_monitor.complete"
