#!/usr/bin/env bash
# Accuracy-first queue: HRNet→A1D→H21 input, H76/H81 RUMPL anchor.
# No MixSTE grafts; no beat-GBT frame matching; skip H112 (VFT bias repeat).
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_accuracy_hrnet_20260805.log

mkdir -p "${ROOT}"
exec 9>"${ROOT}/queue_accuracy_hrnet_20260805.lock"
flock -n 9 || { echo "[accuracy] another queue holds lock; exit"; exit 0; }
exec >>"${LOG}" 2>&1

echo "========================================"
echo "[accuracy] start $(date --iso-8601=seconds)"
echo "[accuracy] input: HRNet coco + A1D + a1d-matched H21"
echo "[accuracy] anchor: H76/H81 centered Plücker + tri-anchor RUMPL"
echo "========================================"

stop_legacy() {
  pkill -f 'launch_MixSTE_alternating_T9_20260804' 2>/dev/null || true
  pkill -f 'MixSTE_alternating_T9_20260804' 2>/dev/null || true
  pkill -f 'fusion-mode mixste-alternating' 2>/dev/null || true
  pkill -f 'queue_beat_gbt_20260805' 2>/dev/null || true
  pkill -f 'H112_H76_VFT_gbtConfGeomBias' 2>/dev/null || true
  sleep 2
  echo "[accuracy] stopped MixSTE-alternating / beat-gbt queue / H112"
}

stop_legacy

run_single() {
  local variant=$1 gpu=$2
  echo "[accuracy] single-frame ${variant} GPU${gpu} $(date --iso-8601=seconds)"
  bash "${AUDIT}/launch_H112_H116_beat_gbt_20260805.sh" "${variant}" "${gpu}"
}

run_temporal() {
  local variant=$1 gpu=$2
  echo "[accuracy] temporal ${variant} GPU${gpu} $(date --iso-8601=seconds)"
  bash "${AUDIT}/launch_H117_H118_temporal_jvt_accuracy_20260805.sh" "${variant}" "${gpu}"
}

# Single-frame (full RUMPL train+eval), parallel where possible
run_single h113_reproj 0 &
p113=$!
run_single h114_v4_train_weight 1 &
p114=$!
wait "${p113}"
wait "${p114}"

run_single h115_global_jv 1 &
p115=$!
run_single h116_h81_reproj 0 &
p116=$!
wait "${p115}"
wait "${p116}"

# Optional multi-frame: H40 global JVT (only if temporal inputs ready)
if [[ -s "${ROOT}/H84_temporal_stride5_validation_inputs/completed.done" ]]; then
  run_temporal h117_frozen_latest 0 &
  p117=$!
  run_temporal h118_unfreeze_vft_latest 1 &
  p118=$!
  wait "${p117}"
  wait "${p118}"
else
  echo "[accuracy] skip H117/H118 — missing ${ROOT}/H84_temporal_stride5_validation_inputs/completed.done"
fi

python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
best = ("H81", root / "H81_H83_targeted_pft/eval/H81_H76_perJointResidualGate_workers12_seed0_20260803")
rows = []
if (best[1] / "V2/table2.json").exists():
    v = {k: json.load(open(best[1]/f"V{k}/table2.json"))["table2_action_equal"]["all17_mm"] for k in (2,3,4)}
    rows.append(("H81 baseline", v))
for base in [
    root / "H112_H116_beat_gbt/eval",
    root / "H117_H118_temporal_jvt_accuracy/eval",
]:
    if not base.exists():
        continue
    for sub in sorted(base.iterdir()):
        if not (sub / "V2/table2.json").exists():
            continue
        vv = {k: json.load(open(sub/f"V{k}/table2.json"))["table2_action_equal"]["all17_mm"] for k in (2,3,4) if (sub/f"V{k}/table2.json").exists()}
        rows.append((sub.name, vv))
print("\n=== accuracy queue summary (All-17 mm) ===")
for name, vv in rows:
    print(name, " ".join(f"V{k}={vv[k]:.3f}" for k in sorted(vv)))
PY

date --iso-8601=seconds >"${ROOT}/queue_accuracy_hrnet_20260805.done"
echo "[accuracy] finished $(date --iso-8601=seconds)"
