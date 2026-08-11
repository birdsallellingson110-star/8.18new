#!/usr/bin/env bash
# Stage 0: full R5 occlusion degradation curve (paper motivation table).
# Pure inference on GPU1 (GPU0 is busy training). Skips already-done combos.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
mkdir -p "$root"

R5="$outbase/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar"
GPU=1

for v in 2 3 4 5; do
  for occ in 0.0 0.3 0.6; do
    tag="R5_v${v}_occ${occ}"
    if [ -f "$root/${tag}_summary.json" ]; then echo "[skip] $tag"; continue; fi
    echo "[gpu${GPU}] $tag $(date +%H:%M:%S)"
    bash "$repo/eval_occlusion_single_20260724.sh" "$GPU" "$tag" "$R5" "$v" baseline "$occ" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag (see $root/${tag}.log)"
  done
done
echo "=== stage0 R5 occlusion full done $(date --iso-8601=seconds) ==="
