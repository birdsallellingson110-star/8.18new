#!/usr/bin/env bash
# Run multi-view V3/V4 experiments two-at-a-time (one job per GPU). HRNet input stack.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_multiview_v34_parallel_20260805.log
LAUNCH=${AUDIT}/launch_H119_H126_multiview_v34_20260805.sh
LAUNCH115=${AUDIT}/launch_H112_H116_beat_gbt_20260805.sh

mkdir -p "${ROOT}"
exec 9>"${ROOT}/queue_multiview_v34_parallel_20260805.lock"
flock -n 9 || { echo "queue already running"; exit 0; }
exec >>"${LOG}" 2>&1

echo "======== multiview V3/V4 queue $(date --iso-8601=seconds) ========"

run_one() {
  local gpu=$1 variant=$2 launcher=$3
  echo "[v34] GPU${gpu} ${variant} start $(date --iso-8601=seconds)"
  export RUMPL_WORKERS=8
  bash "${launcher}" "${variant}" "${gpu}"
  echo "[v34] GPU${gpu} ${variant} done $(date --iso-8601=seconds)"
}

run_pair() {
  local a=$1 ga=$2 b=$3 gb=$4 launcher=$5
  run_one "${ga}" "${a}" "${launcher}" &
  local pa=$!
  run_one "${gb}" "${b}" "${launcher}" &
  local pb=$!
  wait "${pa}" || echo "[v34] WARN ${a} failed"
  wait "${pb}" || echo "[v34] WARN ${b} failed"
}

# Pairs chosen for complementary multi-view hypotheses (train sampling vs fusion vs loss).
run_pair h119_always4 0 h122_relview 1 "${LAUNCH}"
run_pair h123_gjv2 0 h124_mono 1 "${LAUNCH}"
run_pair h125_h81_always4 0 h126_always3 1 "${LAUNCH}"
run_pair h121_h81_w139 0 h120_w139 1 "${LAUNCH}"

# Single-frame helpers from prior queue (skip h112/h114 if still running elsewhere).
for tag_dir in \
  "H113_H76_reprojLambda001_workers12_seed0_20260805" \
  "H114_H76_viewWeights124_workers12_seed0_20260805"; do
  if [[ -s "${ROOT}/H112_H116_beat_gbt/completed/${tag_dir}.done" ]]; then
    echo "[v34] skip ${tag_dir} (already done)"
  else
    echo "[v34] note: ${tag_dir} not done — may overlap with this queue"
  fi
done

run_pair h115_global_jv 0 h116_h81_reproj 1 "${LAUNCH115}"

python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
bases = [
    root / "H81_H83_targeted_pft/eval/H81_H76_perJointResidualGate_workers12_seed0_20260803",
    root / "H76_h50_centered_plucker/eval/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803",
    root / "H119_H126_multiview_v34/eval",
    root / "H112_H116_beat_gbt/eval",
]
print("\n=== V2/V3/V4 All-17 (mm) ===")
def row(name, d):
    if not (d / "V2/table2.json").exists():
        return
    v = {k: json.load(open(d/f"V{k}/table2.json"))["table2_action_equal"]["all17_mm"] for k in (2,3,4)}
    print(f"{name:48s} V2={v[2]:.3f} V3={v.get(3,float('nan')):.3f} V4={v.get(4,float('nan')):.3f}")
for b in bases:
    if b.name == "eval":
        for sub in sorted(b.glob("H*")):
            row(sub.name, sub)
    else:
        row(b.parent.name, b)
PY

date --iso-8601=seconds >"${ROOT}/queue_multiview_v34_parallel_20260805.done"
echo "[v34] finished $(date --iso-8601=seconds)"
