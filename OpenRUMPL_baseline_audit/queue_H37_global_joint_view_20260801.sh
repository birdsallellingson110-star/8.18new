#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

physical_gpu=$1
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/H37_global_joint_view_chain.log
exec >>"${LOG}" 2>&1

echo "[H37 queue] waiting $(date --iso-8601=seconds) physical_gpu=${physical_gpu}"
# H36 is synchronous and covers its three input-only ablations.  Also guard
# against a launcher that may still be finishing evaluation.
while pgrep -f 'queue_H36_paper_inputs_20260801.sh|H36_P[012]_' >/dev/null; do
  sleep 30
done

for variant in plain confgeom; do
  echo "[H37 queue] launch ${variant} $(date --iso-8601=seconds)"
  bash "${AUDIT}/launch_H37_global_joint_view_20260801.sh" "${variant}" "${physical_gpu}"
done

echo "[H37 queue] complete $(date --iso-8601=seconds)"
