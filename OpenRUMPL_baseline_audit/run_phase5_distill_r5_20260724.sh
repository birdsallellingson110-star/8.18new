#!/usr/bin/env bash
# Phase 5: retry distillation on strict R5 baseline (GPU0).
# Trains DS0 general + DS1 hardv+legw0.9, then evaluates V2-V5 all combinations.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/distill_r5_eval
mkdir -p "$root"

train_one() {
  local tag=$1 mode=$2
  local log="$root/${tag}.train.log"
  if find "$outbase" -maxdepth 1 -type d -name "${tag}_*" | grep -q .; then
    echo "[skip-train] $tag (dir exists)"
    return 0
  fi
  echo "[gpu0 train] $tag mode=$mode"
  bash "$repo/run_distill_r5_20260724.sh" 0 "$tag" "$mode" > "$log" 2>&1 \
    || { echo "[FAIL train] $tag"; return 1; }
}

eval_one() {
  local tag=$1
  local ckpt
  ckpt=$(find "$outbase" -maxdepth 2 -type f -path "*${tag}_*/model_best.pth.tar" | sort | tail -1)
  if [ -z "$ckpt" ]; then
    echo "[skip-eval] $tag (no checkpoint)"
    return 1
  fi
  for v in 2 3 4 5; do
    if [ -f "$root/${tag}_v${v}_summary.json" ]; then
      echo "[skip-eval] ${tag}_v${v}"
      continue
    fi
    echo "[gpu0 eval] ${tag}_v${v}"
    bash "$repo/eval_exact_multiview_20260723.sh" 0 "${tag}_v${v}" "$ckpt" "$v" baseline \
      > "$root/${tag}_v${v}.log" 2>&1 \
      || echo "[FAIL eval] ${tag}_v${v}"
  done
}

train_one DS0_general_seed0_20260724 general
eval_one DS0_general_seed0_20260724

train_one DS1_hardv_legw09_seed0_20260724 hardv_legw09
eval_one DS1_hardv_legw09_seed0_20260724

echo "=== phase5 distill R5 done $(date --iso-8601=seconds) ==="
