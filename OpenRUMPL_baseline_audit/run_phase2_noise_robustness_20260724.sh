#!/usr/bin/env bash
# Phase 2: camera-calibration-noise robustness matrix on CMU V2.
# Models: R5 baseline, G2 conf-only, G3 geom-fusion, G4 conf+geom fusion.
# Conditions: clean / mild (rot 5 deg, trans 0.05) / strong (rot 10 deg, trans 0.10).
# Two GPUs used in parallel (model list split).
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
base=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/noise_robustness_eval
mkdir -p "$root"

declare -A CKPT MODE
CKPT[R5]="$base/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar";           MODE[R5]=baseline
CKPT[G2]="$base/G2_gbt_conf_only_exact_seed0_20260723_2026-07-23_08-50-02/model_best.pth.tar";                     MODE[G2]=g2_conf_only
CKPT[G3]="$base/G3_gbt_geom_fusion_only_exact_seed0_20260723_2026-07-23_08-50-02/model_best.pth.tar";              MODE[G3]=g3_geom_fusion
CKPT[G4]="$base/G4_gbt_fusion_geom005_exact_seed0_20260724_2026-07-24_12-04-16/model_best.pth.tar";                MODE[G4]=g4_fusion

# condition_name rot_deg trans_std
conditions=(
  "clean 0 0"
  "mild 5 0.05"
  "strong 10 0.10"
)

run_model_set() {
  local gpu=$1; shift
  local models=("$@")
  for m in "${models[@]}"; do
    for cond in "${conditions[@]}"; do
      read -r cname rot trans <<< "$cond"
      tag="${m}_v2_${cname}"
      if [ -f "$root/${tag}_summary.json" ]; then
        echo "[skip] $tag already done"
        continue
      fi
      echo "[gpu$gpu] $tag  (rot=$rot trans=$trans)"
      bash "$repo/eval_noise_single_20260724.sh" "$gpu" "$tag" "${CKPT[$m]}" 2 "${MODE[$m]}" "$rot" "$trans" \
        > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag (see $root/${tag}.log)"
    done
  done
}

run_model_set 0 R5 G3 &
pid0=$!
run_model_set 1 G2 G4 &
pid1=$!
wait $pid0 $pid1

echo "=== all done, summaries in $root ==="
ls -la "$root"/*_summary.json 2>/dev/null
