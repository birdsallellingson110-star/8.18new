#!/usr/bin/env bash
# H81 single-frame on the same causal-latest temporal protocol as H169 (annot_temporal_5_5).
set -euo pipefail

physical_gpu=${1:-0}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
H81=$(tr -d '\r\n' < "${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt")
OUT=${ROOT}/H169_H174_temporal_h81_gbt_aligned/eval/H81_backbone_only_temporal_protocol
INPUT_DONE=${ROOT}/H84_temporal_stride5_validation_inputs/completed.done

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${REPO}/lib"
export RUMPL_EVAL_STRICT=0
unset RUMPL_N_VIEWS_TRAIN_TEST_ALL

test -s "${INPUT_DONE}"
test -s "${H81}"

mkdir -p "${OUT}/logs"
log="${OUT}/logs/eval.log"
echo "[H81 temporal protocol] start $(date --iso-8601=seconds)" | tee "${log}"

for views in 2 3 4; do
  dest="${OUT}/V${views}"
  mkdir -p "${dest}"
  echo "[H81] V${views} $(date --iso-8601=seconds)" | tee -a "${log}"
  "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${H81}" --backbone-only --backbone-flavor h81 \
    --output-dir "${dest}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 \
    --batch-size 16 --workers 6 --device cuda:0 \
    >>"${dest}/eval.log" 2>&1
  pred=$(find "${dest}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  if [[ -n "${pred}" ]]; then
    "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
      --output-json "${dest}/table2_alt.json" >>"${dest}/table2.log" 2>&1 || true
  fi
done

"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H169_H174_temporal_h81_gbt_aligned/eval")
h81 = root / "H81_backbone_only_temporal_protocol"
h169 = root / "H169_mixsteTTB_mixsteLoss_frozenH81"
print("=== causal-latest protocol (action-equal all17 mm) ===")
for label, base in [("H81 single-frame", h81), ("H169 temporal", h169)]:
    if not base.is_dir():
        continue
    vals = {}
    for v in (2, 3, 4):
        f = base / f"V{v}/table2.json"
        if f.is_file():
            vals[v] = json.load(open(f))["table2_action_equal"]["all17_mm"]
    if vals:
        s = " ".join(f"V{k}={vals[k]:.2f}" for k in sorted(vals))
        print(f"  {label:20s} {s}")
PY | tee -a "${log}"

echo "[H81 temporal protocol] done $(date --iso-8601=seconds)" | tee -a "${log}"
