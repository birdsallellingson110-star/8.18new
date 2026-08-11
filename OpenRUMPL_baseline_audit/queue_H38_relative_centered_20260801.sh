#!/usr/bin/env bash
set -euo pipefail

physical_gpu=${1:-1}
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
BASE=${ROOT}/H38_relative_centered_retained_rumpl
LOG=${ROOT}/H38_relative_centered_chain.log
exec >>"${LOG}" 2>&1

echo "[H38 queue] waiting $(date --iso-8601=seconds) physical_gpu=${physical_gpu}"
while pgrep -f 'queue_H37_global_joint_view_20260801.sh|H37_J[01]_' >/dev/null; do
  sleep 30
done

for variant in relative center; do
  echo "[H38 queue] launch ${variant} $(date --iso-8601=seconds)"
  bash "${AUDIT}/launch_H38_relative_centered_rumpl_20260801.sh" \
    "${variant}" "${physical_gpu}"
done

# Combine only if both independent modules improve H0 on both target metrics.
if /home/lixiaob/cjy/rumpl_venv310/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/mnt/data/cjyoutput/open_source_fusion_audit_20260731')
base = root / 'H38_relative_centered_retained_rumpl/eval'
tags = [
    'H38_R0_relative_A1D_triAnchor_retainedRUMPL_seed0_20260801',
    'H38_C0_center_A1D_triAnchor_retainedRUMPL_seed0_20260801',
]
baseline_dir = root / (
    'H0_a1d_refined_rumpl_tri_anchor/eval/'
    'H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_'
    'fixedK2First8_thenWeighted3to1to1_seed0_20260731'
)
baseline = {
    views: json.load(open(baseline_dir / f'V{views}' / 'table2.json'))[
        'table2_action_equal'
    ]['all17_mm']
    for views in (2, 4)
}
for tag in tags:
    for views in (2, 4):
        path = base / tag / f'V{views}' / 'table2.json'
        value = json.load(open(path))['table2_action_equal']['all17_mm']
        if value >= baseline[views]:
            raise SystemExit(1)
raise SystemExit(0)
PY
then
  echo "[H38 queue] both modules improve V2/V4; launch conditional combination"
  bash "${AUDIT}/launch_H38_relative_centered_rumpl_20260801.sh" \
    relative_center "${physical_gpu}"
else
  echo "[H38 queue] combination skipped: both modules did not independently improve V2/V4"
fi

"${AUDIT}/summarize_h36m_accuracy_assault.py"
echo "[H38 queue] complete $(date --iso-8601=seconds)"
