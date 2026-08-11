#!/usr/bin/env bash
# Continue accuracy assault without GBT bias (H32 proved bias hurts on A1D).
set -euo pipefail

GPU=${1:-1}
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/H33_H34_nobias_chain.log
SCOREBOARD=${ROOT}/ACCURACY_ASSAULT_SCOREBOARD.txt
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python

exec >>"${LOG}" 2>&1
echo "[nobias] start $(date --iso-8601=seconds) gpu=${GPU}"

refresh() {
  {
    echo "=== Accuracy assault scoreboard ($(date --iso-8601=seconds)) ==="
    echo "TARGETS: V2 < 40 | V4 < 30"
    echo "BEST so far H0: V2=55.700 V4=37.585 | H32+bias REGRESSED to 59.5/40.0"
    echo
    "${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
jobs = [
    ("H0_a1d_tri", root/"H0_a1d_refined_rumpl_tri_anchor/eval/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731"),
]
for base in [
    root/"H32_tri_anchor_gbt_bias/eval",
    root/"H33_ultra_v2_a1d_nobias/eval",
    root/"H34_a1d_nobias_geom_losses/eval",
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
best_v2, best_v4, best_name = 1e9, 1e9, None
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
        if v2 < best_v2:
            best_v2, best_v4, best_name = v2, v4, name
    print(f"{name:72s} {fmt(vals[0])} {fmt(vals[1])} {fmt(vals[2])}  {hit}")
if best_name:
    print(f"\nBEST: {best_name} V2={best_v2:.3f} V4={best_v4:.3f}")
PY
  } | tee "${SCOREBOARD}"
}

echo "[nobias] launch H33 ultra-V2 no-bias"
bash "${AUDIT}/launch_H33_ultra_v2_a1d_nobias_20260731.sh" "${GPU}"
refresh

echo "[nobias] launch H34 geom-losses no-bias"
bash "${AUDIT}/launch_H34_a1d_nobias_geom_losses_20260731.sh" "${GPU}"
refresh

echo "[nobias] complete $(date --iso-8601=seconds)"
