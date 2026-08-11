#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
log="$root/phase0_j2_s1_eval_20260724.log"
report="$root/phase0_j2_s1_eval_summary_20260724.txt"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python

j2_ckpt="$root/output/multiview_amass_rumpl/multiview_rumpl_999/J2_adaptive_globalJV_d2_nobias_g005_012_seed0_20260723_2026-07-23_20-59-32/final_state.pth.tar"
s1_ckpt="$root/output/multiview_amass_rumpl/multiview_rumpl_999/S1_rumpl_symmetry_w05_seed0_20260723_2026-07-23_21-19-01/final_state.pth.tar"

exec > >(tee -a "$log") 2>&1
echo "START $(date --iso-8601=seconds)"

for ckpt in "$j2_ckpt" "$s1_ckpt"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

eval_family() {
  local gpu=$1 family=$2 checkpoint=$3 mode=$4
  for n_views in 2 3 4 5; do
    echo "=== eval $family V$n_views on GPU$gpu ==="
    "$repo/eval_exact_multiview_20260723.sh" \
      "$gpu" "${family}_v${n_views}" "$checkpoint" "$n_views" "$mode"
  done
}

eval_family 0 J2 "$j2_ckpt" j2_adaptive &
pid_j2=$!
eval_family 1 S1 "$s1_ckpt" baseline &
pid_s1=$!

wait "$pid_j2"
wait "$pid_s1"

"$python" "$repo/summarize_phase0_j2_s1_20260724.py" | tee "$report"
echo "REPORT $report"
echo "END $(date --iso-8601=seconds)"
