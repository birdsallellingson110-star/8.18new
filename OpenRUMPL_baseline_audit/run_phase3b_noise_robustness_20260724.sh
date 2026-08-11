#!/usr/bin/env bash
# Phase 3b: camera-calibration-noise robustness on CMU V2.
# Tests whether depth-aux-trained model (D2) degrades less than baseline (R5) under noise.
# Pure inference on GPU0. D1 (tri-anchor) included to see if the geometric anchor helps under noise.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/noise_robustness_eval
mkdir -p "$root"

R5="$outbase/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar"
D1=$(find "$outbase" -maxdepth 1 -type d -name 'D1_tri_anchor_seed0_20260724_*' | sort | tail -1)/model_best.pth.tar
D2=$(find "$outbase" -maxdepth 1 -type d -name 'D2_ray_depth_aux_w01_seed0_20260724_*' | sort | tail -1)/model_best.pth.tar

declare -A CKPT MODE
CKPT[R5]="$R5"; MODE[R5]=baseline
CKPT[D2]="$D2"; MODE[D2]=d2_depthaux
CKPT[D1]="$D1"; MODE[D1]=d1_anchor

# name rot_deg trans_std
conditions=(
  "clean 0 0"
  "mild 3 0.03"
  "strong 8 0.08"
)

for m in R5 D2 D1; do
  for cond in "${conditions[@]}"; do
    read -r cname rot trans <<< "$cond"
    tag="${m}_v2_${cname}"
    if [ -f "$root/${tag}_summary.json" ]; then
      echo "[skip] $tag done"; continue
    fi
    echo "[gpu0] $tag rot=$rot trans=$trans"
    bash "$repo/eval_noise_single_20260724.sh" 0 "$tag" "${CKPT[$m]}" 2 "${MODE[$m]}" "$rot" "$trans" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag (see $root/${tag}.log)"
  done
done

echo "=== phase3b done $(date --iso-8601=seconds) ==="
