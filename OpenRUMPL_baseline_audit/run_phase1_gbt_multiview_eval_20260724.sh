#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
out_root="$root/gbt_multiview_eval"
log="$root/phase1_gbt_multiview_eval_20260724.log"
report="$root/phase1_gbt_multiview_eval_summary_20260724.txt"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
model_root="$root/output/multiview_amass_rumpl/multiview_rumpl_999"

exec > >(tee -a "$log") 2>&1
echo "START $(date --iso-8601=seconds)"

find_ckpt() {
  find "$model_root" -maxdepth 1 -type d -name "$1_*" -print | sort | tail -n 1
}

eval_one() {
  local gpu=$1 variant=$2 mode=$3
  local dir
  dir=$(find_ckpt "$variant")
  local ckpt="$dir/model_best.pth.tar"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$dir/final_state.pth.tar"
  fi
  echo "=== $variant mode=$mode ckpt=$ckpt ==="
  "$repo/eval_exact_gbt_multiview_20260724.sh" "$gpu" "$variant" "$ckpt" "$mode"
}

eval_one 0 G1_gbt_fusion_exact_seed0_20260723 g1_fusion &
pid_g1=$!
eval_one 1 G3_gbt_geom_fusion_only_exact_seed0_20260723 g3_geom_fusion &
pid_g3=$!
wait "$pid_g1" "$pid_g3"

eval_one 0 G2_gbt_conf_only_exact_seed0_20260723 g2_conf_only &
pid_g2=$!
eval_one 1 G0_gbt_formula_exact_seed0_20260723 g0_formula &
pid_g0=$!
wait "$pid_g2" "$pid_g0"

"$python" "$repo/summarize_phase1_gbt_multiview_20260724.py" | tee "$report"
echo "REPORT $report"
echo "END $(date --iso-8601=seconds)"
