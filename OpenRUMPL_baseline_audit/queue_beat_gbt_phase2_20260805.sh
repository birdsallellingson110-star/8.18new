#!/usr/bin/env bash
# Phase-2 beat-GBT queue (GPU0). Targets: V2<36.8 V3<30.4 V4<26.0 (GBT Table-I, T=9 ref).
# H81 single-frame baseline: 34.61 / 30.81 / 29.99 — prioritize temporal + geometry fusion.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_beat_gbt_phase2_20260805.log
LAUNCH=${AUDIT}/launch_H135_H140_beat_gbt_phase2_20260805.sh
SCORE=${ROOT}/BEAT_GBT_SCOREBOARD_phase2.txt

mkdir -p "${ROOT}"
exec 9>"${ROOT}/queue_beat_gbt_phase2_20260805.lock"
flock -n 9 || { echo "[phase2] already running"; exit 0; }
exec >>"${LOG}" 2>&1

echo "======== beat-gbt phase2 $(date --iso-8601=seconds) ========"
echo "GBT targets (All-17 mm): V2<36.8 V3<30.4 V4<26.0"
echo "H81 single-frame: 34.61 / 30.81 / 29.99"
echo "H112 GBT-bias on H76 REGRESSED — skip repeat; H140 tries H81 stack only"

run() {
  local var=$1
  echo "[phase2] GPU0 ${var} $(date --iso-8601=seconds)"
  bash "${LAUNCH}" "${var}" 0
  score
}

score() {
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
targets = {2: 36.8, 3: 30.4, 4: 26.0}
rows = []
for label, ev in [
    ("H81_sf", root / "H81_H83_targeted_pft/eval/H81_H76_perJointResidualGate_workers12_seed0_20260803"),
]:
    if not (ev / "V2/table2.json").exists():
        continue
    vals = {v: json.load(open(ev / f"V{v}/table2.json"))["table2_action_equal"]["all17_mm"] for v in (2,3,4)}
    rows.append((label, vals))
for ev in sorted((root / "H135_H140_beat_gbt_phase2/eval").glob("*")):
    if not (ev / "V2/table2.json").exists():
        continue
    vals = {v: json.load(open(ev / f"V{v}/table2.json"))["table2_action_equal"]["all17_mm"] for v in (2,3,4) if (ev/f"V{v}/table2.json").exists()}
    if 2 in vals:
        rows.append((ev.name[:40], vals))
for base in [root / "H112_H116_beat_gbt/eval", root / "H135_H140_beat_gbt_phase2/eval"]:
    if not base.exists():
        continue
    for sub in sorted(base.glob("*")):
        if not (sub / "V2/table2.json").exists():
            continue
        vals = {v: json.load(open(sub / f"V{v}/table2.json"))["table2_action_equal"]["all17_mm"] for v in (2,3,4)}
        rows.append((sub.name[:40], vals))
print("\n=== scoreboard All-17 action-equal (mm) ===")
best = None
for label, vals in rows:
    beat = all(vals.get(k, 999) < targets[k] for k in (2,3,4) if k in vals)
    s = " ".join(f"V{k}={vals[k]:.2f}" for k in sorted(vals))
    mark = " *** BEAT GBT ***" if beat else ""
    print(f"  {label:42s} {s}{mark}")
    if beat and (best is None or vals.get(4,999) < best[1].get(4,999)):
        best = (label, vals)
if best:
    print(f"\nBest full beat: {best[0]}")
PY
}

# Order: temporal first (main GBT lever), then geometry fusion, then train weights
run h135_temporal_h81
run h136_temporal_h81_unfreeze
run h137_h81_depro_caa
run h138_h114_v4w
run h139_h117_temporal_h76
run h140_h81_gbt_bias_ft

date --iso-8601=seconds >"${ROOT}/queue_beat_gbt_phase2_20260805.done"
echo "[phase2] finished $(date --iso-8601=seconds)"
