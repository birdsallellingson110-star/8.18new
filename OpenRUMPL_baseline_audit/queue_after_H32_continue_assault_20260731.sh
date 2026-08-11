#!/usr/bin/env bash
# After H32(a1d+both) finishes: refresh scoreboard, run H33 ultra-V2, then H34 geom-losses.
set -euo pipefail

GPU=${1:-1}
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
H32_DONE=${ROOT}/H32_tri_anchor_gbt_bias/completed/H32_a1dRefined2D_confGeom_triAnchor_curriculum_seed0_20260731.done
LOG=${ROOT}/after_H32_continue.log
SCOREBOARD=${ROOT}/ACCURACY_ASSAULT_SCOREBOARD.txt
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python

exec >>"${LOG}" 2>&1
echo "[continue] start $(date --iso-8601=seconds) waiting H32"

while [[ ! -s "${H32_DONE}" ]]; do sleep 60; done
echo "[continue] H32 done $(date --iso-8601=seconds)"

# Stop the older assault script so it does not burn the GPU on conf/geom
# ablations before H33/H34 (higher-leverage next steps).
pkill -f "queue_accuracy_assault_after_H0_20260731.sh" 2>/dev/null || true
# If an H32 conf/geom child just spawned, stop those too (keep completed both).
pkill -f "launch_H32_tri_anchor_gbt_bias_20260731.sh a1d ${GPU} conf" 2>/dev/null || true
pkill -f "launch_H32_tri_anchor_gbt_bias_20260731.sh a1d ${GPU} geom" 2>/dev/null || true
sleep 2
echo "[continue] redirected GPU to H33/H34"

refresh() {
  {
    echo "=== Accuracy assault scoreboard ($(date --iso-8601=seconds)) ==="
    echo "TARGETS: V2 All < 40.0 | V4 All < 30.0"
    echo "H22: 62.268 / 38.333 | H31b: 61.294 / 37.692 | H0: 55.700 / 37.585"
    echo
    "${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
jobs = [
    ("H0_a1d_tri", root/"H0_a1d_refined_rumpl_tri_anchor/eval/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731"),
]
for base, name in [
    (root/"H32_tri_anchor_gbt_bias/eval", None),
    (root/"H33_ultra_v2_a1d_bias/eval", None),
    (root/"H34_a1d_bias_geom_losses/eval", None),
]:
    if base.is_dir():
        for p in sorted(base.iterdir()):
            jobs.append((p.name, p))

def read(eval_root, n):
    p = eval_root / f"V{n}" / "table2.json"
    if not p.is_file():
        return None
    t = json.load(open(p))["table2_action_equal"]
    return t["all17_mm"], t.get("kp_star_mm")

print(f"{'tag':72s} {'V2':>8s} {'V3':>8s} {'V4':>8s}  hit?")
for name, path in jobs:
    vals = [read(path, n) for n in (2, 3, 4)]
    if all(v is None for v in vals):
        continue
    def fmt(r):
        return f"{r[0]:8.3f}" if r else f"{'--':>8s}"
    v2 = vals[0][0] if vals[0] else None
    v4 = vals[2][0] if vals[2] else None
    hit = ""
    if v2 is not None and v4 is not None:
        hit = "HIT" if v2 < 40 and v4 < 30 else (
            "V2ok" if v2 < 40 else ("V4ok" if v4 < 30 else "MISS"))
    print(f"{name:72s} {fmt(vals[0])} {fmt(vals[1])} {fmt(vals[2])}  {hit}")
PY
  } | tee "${SCOREBOARD}"
}

refresh

echo "[continue] launch H33 ultra-V2 $(date --iso-8601=seconds)"
bash "${AUDIT}/launch_H33_ultra_v2_a1d_bias_20260731.sh" "${GPU}"
refresh

echo "[continue] launch H34 geom-losses $(date --iso-8601=seconds)"
bash "${AUDIT}/launch_H34_a1d_bias_geom_losses_20260731.sh" "${GPU}"
refresh

echo "[continue] complete $(date --iso-8601=seconds)"
