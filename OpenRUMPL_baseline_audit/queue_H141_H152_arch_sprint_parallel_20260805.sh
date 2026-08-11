#!/usr/bin/env bash
# Run H141-H152 two-at-a-time (one job per GPU). Re-run safe (skip registry + completed).
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LAUNCH=${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh
LOG=${ROOT}/H141_H152_arch_sprint/queue_parallel.log

mkdir -p "${ROOT}/H141_H152_arch_sprint"
exec 9>"${ROOT}/H141_H152_arch_sprint/queue.lock"
flock -n 9 || { echo "arch sprint queue already running"; exit 0; }
exec >>"${LOG}" 2>&1

# shellcheck source=/dev/null
source "${AUDIT}/experiment_should_skip.sh"

echo "======== arch sprint queue $(date --iso-8601=seconds) ========"

run_one() {
  local gpu=$1 var=$2
  if experiment_should_skip_variant "${var}" 2>/dev/null; then
    echo "[arch] skip ${var} (alias)"
    return 0
  fi
  echo "[arch] GPU${gpu} ${var} start $(date --iso-8601=seconds)"
  bash "${LAUNCH}" "${var}" "${gpu}" || echo "[arch] WARN ${var} failed"
  echo "[arch] GPU${gpu} ${var} done $(date --iso-8601=seconds)"
}

run_pair() {
  run_one 0 "$1" &
  local pa=$!
  run_one 1 "$2" &
  local pb=$!
  wait "${pa}" || true
  wait "${pb}" || true
}

VARIANTS=(
  h141_no_pft_repeat
  h142_relview_w322
  h143_gjv2_h81
  h144_graph_res_h81
  h145_no_tri_anchor
  h146_set_decoder_h76
  h147_vft_mask04
  h148_jv1_biased_h81
  h149_gate_gjv2_w322
  h150_gate_relview_w322
  h151_bone_ray01
  h152_shallow_vft
)

i=0
while [[ $i -lt ${#VARIANTS[@]} ]]; do
  a=${VARIANTS[$i]}
  b=${VARIANTS[$((i + 1))]:-}
  if [[ -z "${b}" ]]; then
    run_one 0 "${a}"
  else
    run_pair "${a}" "${b}"
  fi
  i=$((i + 2))
done

python3 "${AUDIT}/scan_experiment_skip_registry_20260805.py" || true

python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
targets = {"V2": 36.8, "V3": 30.4, "V4": 26.0}
h81 = root / "H81_H83_targeted_pft/eval/H81_H76_perJointResidualGate_workers12_seed0_20260803"
base = root / "H141_H152_arch_sprint/eval"
print("\n=== H141-H152 vs GBT / H81 (All-17 mm) ===")
for sub in sorted(base.glob("H14*")) if base.is_dir() else []:
    vals = {}
    for v in (2, 3, 4):
        f = sub / f"V{v}/table2.json"
        if f.is_file():
            vals[v] = json.load(open(f))["table2_action_equal"]["all17_mm"]
    if 2 not in vals:
        continue
    beat = sum(1 for k in (2, 3, 4) if k in vals and vals[k] < targets[f"V{k}"])
    s = " ".join(f"V{k}={vals[k]:.2f}" for k in sorted(vals))
    print(f"  {sub.name[:28]:28s} {s}  beat={beat}/3")
PY

echo "[arch] finished $(date --iso-8601=seconds)"
