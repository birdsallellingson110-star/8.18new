#!/usr/bin/env bash
# Start E2 and H18 only after the repaired, predeclared HRNet endpoint matches
# the old C2 generator on every formal view-count metric.  All artifacts use a
# fresh root so the stale epoch-7 cache cannot be reused.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
PHASE1=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/phase1_cont20_lr1e5
NEW_ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/downstream_phase1
LIB=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_downstream_20260824.sh

while [[ ! -s "${PHASE1}/COMPLETED" ]]; do sleep 30; done
mkdir -p "${NEW_ROOT}/hrnet/generator"

"${PY}" - "${PHASE1}" "${NEW_ROOT}/generator_gate.json" <<'PY'
import json
import pathlib
import sys

phase = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
old_c2 = {2: 38.686, 3: 30.943, 4: 28.629}
measured = {}
for views in (2, 3, 4):
    source = phase / 'eval' / f'V{views}' / 'table2.json'
    with source.open() as stream:
        payload = json.load(stream)
    measured[views] = float(payload['table2_action_equal']['all17_mm'])

passed = all(measured[views] <= old_c2[views] for views in old_c2)
report = {
    'selection_policy': 'predeclared final epoch; formal metrics are gate-only',
    'old_c2_mm': {f'V{k}': v for k, v in old_c2.items()},
    'repaired_mm': {f'V{k}': v for k, v in measured.items()},
    'delta_mm': {f'V{k}': measured[k] - old_c2[k] for k in old_c2},
    'passed_all_metrics': passed,
}
with output.open('w') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report, indent=2))
if not passed:
    raise SystemExit(42)
PY

checkpoint=$(cat "${PHASE1}/final_checkpoint.txt")
test -s "${checkpoint}"
printf '%s\n' "${checkpoint}" >"${NEW_ROOT}/hrnet/generator/checkpoint.txt"
{
  echo "source_phase=${PHASE1}"
  echo "checkpoint=${checkpoint}"
  echo "cache_policy=fresh root; no reuse from stage1_h36m_dual_frontend"
} >"${NEW_ROOT}/hrnet/generator/manifest.txt"

# Source the shared, frozen E2/H18 implementation without invoking its normal
# two-frontend main function, then run only the repaired HRNet branch.
source "${LIB}"
ROOT="${NEW_ROOT}"
run_frontend hrnet gbt_yolox_x_score001_fallback_legswap 0 \
  /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
  /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl \
  annot_temporal_5_5 \
  /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl

date --iso-8601=seconds >"${NEW_ROOT}/STAGE1_COMPLETED"
echo "[repaired HRNet E2/H18] complete ${NEW_ROOT}"
