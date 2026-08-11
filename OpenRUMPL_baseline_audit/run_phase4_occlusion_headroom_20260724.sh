#!/usr/bin/env bash
# Phase 4: establish RUMPL degradation curve under structured 2D occlusion (the correct robustness axis).
# Pure inference on GPU0. R5 baseline + D2 depth-aux under occlusion 0 / 0.3 / 0.6 on V2 and V3.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
mkdir -p "$root"

R5="$outbase/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar"
D2=$(find "$outbase" -maxdepth 1 -type d -name 'D2_ray_depth_aux_w01_seed0_20260724_*' | sort | tail -1)/model_best.pth.tar

declare -A CKPT MODE
CKPT[R5]="$R5"; MODE[R5]=baseline
CKPT[D2]="$D2"; MODE[D2]=d2_depthaux

for m in R5 D2; do
  for v in 2 3; do
    for occ in 0.0 0.3 0.6; do
      tag="${m}_v${v}_occ${occ}"
      if [ -f "$root/${tag}_summary.json" ]; then echo "[skip] $tag"; continue; fi
      echo "[gpu0] $tag"
      bash "$repo/eval_occlusion_single_20260724.sh" 0 "$tag" "${CKPT[$m]}" "$v" "${MODE[$m]}" "$occ" \
        > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag (see $root/${tag}.log)"
    done
  done
done
echo "=== phase4 occlusion headroom done $(date --iso-8601=seconds) ==="
