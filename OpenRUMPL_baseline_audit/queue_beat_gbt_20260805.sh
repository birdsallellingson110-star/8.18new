#!/usr/bin/env bash
# Beat GBT-HRNet (Table-2 absolute All-17): V2<36.8, V3<30.4, V4<26.0.
# Current best single-frame ~H81: 34.6/30.8/30.0 — V4 is the main gap.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_beat_gbt_20260805.log
SCORE=${ROOT}/BEAT_GBT_SCOREBOARD.txt

mkdir -p "${ROOT}"
exec > >(tee -a "${LOG}") 2>&1

echo "========================================"
echo "[beat-gbt] start $(date --iso-8601=seconds)"
echo "[beat-gbt] paper target (GBT-HRNet, 9-frame): V2<36.8 V3<30.4 V4<26.0"
echo "[beat-gbt] our H81 single-frame: ~34.61/30.81/29.99"
echo "========================================"

append_scores() {
  "${AUDIT}/collect_master_experiment_ledger_20260801.py" \
    --tags H112,H113,H114,H115,H116 \
    --output "${SCORE}" 2>/dev/null || true
  if [[ -s "${SCORE}" ]]; then
    echo "[beat-gbt] scoreboard:"
    tail -20 "${SCORE}"
  fi
}

echo "[beat-gbt] phase0: J1/J2 fast eval (GPU0)"
bash "${AUDIT}/eval_MixSTE_joint_pft_T9_fast_20260804.sh"

echo "[beat-gbt] phase1: MixSTE alternating T=9 train+eval (GPU0, background)"
nohup bash "${AUDIT}/launch_MixSTE_alternating_T9_20260804.sh" \
  >"${ROOT}/MixSTE_alternating_T9_20260805.nohup.log" 2>&1 &
alt_pid=$!
echo "[beat-gbt] MixSTE alternating pid=${alt_pid}"

run_rumpl() {
  local variant=$1
  echo "[beat-gbt] RUMPL ${variant} on GPU1 $(date --iso-8601=seconds)"
  bash "${AUDIT}/launch_H112_H116_beat_gbt_20260805.sh" "${variant}" 1
  append_scores
}

echo "[beat-gbt] phase2: H112-H116 sequential on GPU1"
run_rumpl h112_vft_bias
run_rumpl h113_reproj
run_rumpl h114_v4_train_weight
run_rumpl h115_global_jv
run_rumpl h116_h81_reproj

echo "[beat-gbt] waiting MixSTE alternating (pid=${alt_pid})"
wait "${alt_pid}" || echo "[beat-gbt] alternating exited non-zero — see nohup log"

echo "[beat-gbt] MixSTE alternating done $(date --iso-8601=seconds)"
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
targets = {"V2": 36.8, "V3": 30.4, "V4": 26.0}
rows = []
for label, base in [
    ("H81", root / "H81_H83_targeted_pft/eval/H81_H76_perJointResidualGate_workers12_seed0_20260803"),
    ("H76", root / "H76_h50_centered_plucker/eval/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803"),
    ("MixSTE_M0", root / "MixSTE_T9_strict_20260804/eval_fast/M0_h76_zero_ttb"),
    ("MixSTE_alt_A1", root / "MixSTE_alternating_T9_20260804/eval/A1_alternating_rumpl_mpjpe"),
    ("J1", root / "MixSTE_joint_pft_T9_20260804/eval_fast/J1_ttb_pft_rumpl_mpjpe"),
]:
    d = base
    if not (d / "V2/table2.json").exists():
        continue
    vals = {}
    for v in (2, 3, 4):
        f = d / f"V{v}/table2.json"
        if f.exists():
            vals[v] = json.load(open(f))["table2_action_equal"]["all17_mm"]
    if 2 not in vals:
        continue
    beat = all(vals.get(k, 999) < targets[f"V{k}"] for k in (2, 3, 4) if k in vals)
    rows.append((label, vals, beat))
print("\n=== beat GBT-HRNet check (All-17 mm) ===")
for label, vals, beat in rows:
    s = " ".join(f"V{k}={vals[k]:.3f}" for k in sorted(vals))
    print(f"  {label:16s} {s}  {'BEAT' if beat else 'gap'}")
for tag in ["H112", "H113", "H114", "H115", "H116"]:
    ev = root / "H112_H116_beat_gbt/eval"
    if not ev.exists():
        continue
    for sub in sorted(ev.glob(f"{tag}_*")):
        if not (sub / "V2/table2.json").exists():
            continue
        vals = {v: json.load(open(sub / f"V{v}/table2.json"))["table2_action_equal"]["all17_mm"] for v in (2,3,4) if (sub/f"V{v}/table2.json").exists()}
        beat = all(vals.get(k, 999) < targets[f"V{k}"] for k in (2, 3, 4))
        s = " ".join(f"V{k}={vals[k]:.3f}" for k in sorted(vals))
        print(f"  {sub.name:16s} {s}  {'BEAT' if beat else 'gap'}")
PY

date --iso-8601=seconds >"${ROOT}/queue_beat_gbt_20260805.done"
echo "[beat-gbt] finished $(date --iso-8601=seconds)"
