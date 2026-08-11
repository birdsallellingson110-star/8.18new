#!/usr/bin/env bash
# Accuracy assault chain (targets: V2<40, V4<30).
# Waits for H0 curriculum done, then launches successive H32 variants on GPU.
set -euo pipefail

GPU=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
H0_DONE=${ROOT}/H0_a1d_refined_rumpl_tri_anchor/completed/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731.done
H0_TRAIN_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_legswap/h36m_train.pkl
SCOREBOARD=${ROOT}/ACCURACY_ASSAULT_SCOREBOARD.txt
LOG=${ROOT}/accuracy_assault_chain.log

exec >>"${LOG}" 2>&1
echo "[assault] start $(date --iso-8601=seconds) gpu=${GPU}"

echo "[assault] wait H0 train PKL..."
while [[ ! -s "${H0_TRAIN_PKL}" ]]; do sleep 30; done
echo "[assault] wait H0 done file..."
while [[ ! -s "${H0_DONE}" ]]; do sleep 60; done
echo "[assault] H0 complete $(date --iso-8601=seconds)"

write_scoreboard() {
  {
    echo "=== Accuracy assault scoreboard ($(date --iso-8601=seconds)) ==="
    echo "TARGETS: V2 All < 40.0 | V4 All < 30.0"
    echo "baseline H22: V2=62.268 V4=38.333"
    echo "best gate H31b: V2=61.294 V4=37.692"
    echo
    "${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
jobs = []
# H0
h0 = root / "H0_a1d_refined_rumpl_tri_anchor/eval/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731"
jobs.append(("H0_a1d_tri", h0))
# H32 variants
h32 = root / "H32_tri_anchor_gbt_bias/eval"
if h32.is_dir():
    for p in sorted(h32.iterdir()):
        jobs.append((p.name, p))

def read_table2(eval_root, n):
    p = eval_root / f"V{n}" / "table2.json"
    if not p.is_file():
        return None
    d = json.load(open(p))
    t = d.get("table2_action_equal", {})
    return t.get("all17_mm"), t.get("kp_star_mm")

print(f"{'tag':70s} {'V2':>8s} {'V3':>8s} {'V4':>8s}  hit?")
for name, path in jobs:
    vals = []
    for n in (2, 3, 4):
        r = read_table2(path, n)
        vals.append(r)
    if all(v is None for v in vals):
        continue
    def fmt(r):
        return f"{r[0]:8.3f}" if r and r[0] is not None else f"{'--':>8s}"
    v2 = vals[0][0] if vals[0] else None
    v4 = vals[2][0] if vals[2] else None
    hit = ""
    if v2 is not None and v4 is not None:
        hit = ("HIT" if v2 < 40 and v4 < 30 else
               ("V2ok" if v2 < 40 else ("V4ok" if v4 < 30 else "MISS")))
    print(f"{name:70s} {fmt(vals[0])} {fmt(vals[1])} {fmt(vals[2])}  {hit}")
PY
  } >"${SCOREBOARD}"
  cat "${SCOREBOARD}"
}

# Always refresh scoreboard with H0 first.
write_scoreboard

# Decision helper: if H0 already hits targets, still run H32 for further push;
# if H0 misses badly on V2, still push A1D+bias as primary next lever.
echo "[assault] launch H32 a1d+confGeom $(date --iso-8601=seconds)"
bash "${AUDIT}/launch_H32_tri_anchor_gbt_bias_20260731.sh" a1d "${GPU}" both
write_scoreboard

# If still missing V2, try conf-only and geom-only ablations to isolate which bias helps.
v2=$("${PY}" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H32_tri_anchor_gbt_bias/eval")
cands=sorted(p.glob("H32_a1dRefined2D_confGeom_*/V2/table2.json"))
if not cands:
    print(999)
else:
    d=json.load(open(cands[-1]))
    print(d["table2_action_equal"]["all17_mm"])
PY
)
echo "[assault] H32 a1d+both V2=${v2}"

if awk "BEGIN{exit !(${v2} >= 40)}"; then
  echo "[assault] V2 still >=40 → try conf-only then geom-only"
  bash "${AUDIT}/launch_H32_tri_anchor_gbt_bias_20260731.sh" a1d "${GPU}" conf
  write_scoreboard
  bash "${AUDIT}/launch_H32_tri_anchor_gbt_bias_20260731.sh" a1d "${GPU}" geom
  write_scoreboard
fi

# If V4 still >=30 after a1d+both, keep going with original2D+bias control for diagnosis.
v4=$("${PY}" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H32_tri_anchor_gbt_bias/eval")
cands=sorted(p.glob("H32_a1dRefined2D_confGeom_*/V4/table2.json"))
if not cands:
    print(999)
else:
    d=json.load(open(cands[-1]))
    print(d["table2_action_equal"]["all17_mm"])
PY
)
echo "[assault] H32 a1d+both V4=${v4}"

echo "[assault] complete $(date --iso-8601=seconds)"
write_scoreboard
