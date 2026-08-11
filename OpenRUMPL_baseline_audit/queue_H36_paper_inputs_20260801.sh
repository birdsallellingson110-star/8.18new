#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

physical_gpu=$1
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/H36_paper_input_chain.log

exec >>"${LOG}" 2>&1
echo "[H36 queue] waiting $(date --iso-8601=seconds) physical_gpu=${physical_gpu}"

# H33 launches H34, and the H35 queue then exports refined PKLs and trains on
# the same physical GPU.  Do not compete with any part of that chain.
while pgrep -f 'H33_a1d_nobias_triAnchor|H34_a1d_nobias_triAnchor|H35_a1dH21_nobias_triAnchor|export_h21_refined_mmpose_pkl.py|queue_H35_after_H34_20260731.sh' >/dev/null; do
  sleep 30
done

for variant in plucker harmonic15; do
  echo "[H36 queue] launch ${variant} $(date --iso-8601=seconds)"
  bash "${AUDIT}/launch_H36_paper_input_rumpl_20260801.sh" "${variant}" "${physical_gpu}"
done

# Do not spend a full run on the combined representation unless both isolated
# input changes are independently useful under the fixed H0 protocol.
if /home/lixiaob/cjy/rumpl_venv310/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/data/cjyoutput/open_source_fusion_audit_20260731')
baseline_dir = root / (
    'H0_a1d_refined_rumpl_tri_anchor/eval/'
    'H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_'
    'fixedK2First8_thenWeighted3to1to1_seed0_20260731'
)
base = {
    views: json.load(open(baseline_dir / f'V{views}' / 'table2.json'))[
        'table2_action_equal'
    ]['all17_mm']
    for views in (2, 3, 4)
}
tags = [
    'H36_P0_plucker_A1D_triAnchor_retainedRUMPL_curriculum_seed0_20260801',
    'H36_P1_harmonic15_A1D_triAnchor_retainedRUMPL_curriculum_seed0_20260801',
]
eval_root = root / 'H36_paper_input_retained_rumpl/eval'
for tag in tags:
    values = {
        views: json.load(open(eval_root / tag / f'V{views}' / 'table2.json'))[
            'table2_action_equal'
        ]['all17_mm']
        for views in (2, 3, 4)
    }
    target_gain = max(base[2] - values[2], base[4] - values[4])
    worst_regression = max(values[views] - base[views] for views in (2, 3, 4))
    if target_gain < 1.0 or worst_regression > 0.5:
        raise SystemExit(1)
raise SystemExit(0)
PY
then
  echo "[H36 queue] both inputs pass gate; launch plucker_harmonic15"
  bash "${AUDIT}/launch_H36_paper_input_rumpl_20260801.sh" \
    plucker_harmonic15 "${physical_gpu}"
else
  echo "[H36 queue] combination skipped: isolated inputs did not both pass gate"
fi

echo "[H36 queue] complete $(date --iso-8601=seconds)"
