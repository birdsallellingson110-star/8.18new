#!/usr/bin/env bash
# Evaluate the Stage 2a (structured-occlusion-trained) checkpoint on the full
# occlusion axis: V2-V5 x occ{0,0.3,0.6}. Architecture is unchanged vs R5, so mode=baseline.
# Pure inference. Default GPU 1. Skips done combos.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
GPU=${1:-1}
variant=${2:-S2a_structocc_L04_seed0_20260724}

ckpt=$(find "$outbase" -maxdepth 1 -type d -name "${variant}_*" | sort | tail -1)/model_best.pth.tar
if [ ! -f "$ckpt" ]; then echo "checkpoint not found for $variant: $ckpt" >&2; exit 1; fi
echo "using checkpoint: $ckpt"

for v in 2 3 4 5; do
  for occ in 0.0 0.3 0.6; do
    tag="S2a_v${v}_occ${occ}"
    if [ -f "$root/${tag}_summary.json" ]; then echo "[skip] $tag"; continue; fi
    echo "[gpu${GPU}] $tag $(date +%H:%M:%S)"
    bash "$repo/eval_occlusion_single_20260724.sh" "$GPU" "$tag" "$ckpt" "$v" baseline "$occ" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag (see $root/${tag}.log)"
  done
done
echo "=== stage2a occlusion eval done $(date --iso-8601=seconds) ==="
