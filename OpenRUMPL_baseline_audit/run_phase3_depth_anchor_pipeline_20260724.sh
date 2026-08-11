#!/usr/bin/env bash
# Phase 3: literature-backed modules, two GPUs in parallel.
#   GPU0: D1 tri-anchor residual  -> eval V2-V5 -> D3 anchor+depthaux combo -> eval
#   GPU1: D2 ray-depth aux w=0.1  -> eval V2-V5 -> D4 ray-depth aux w=0.3   -> eval
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722

find_ckpt() {
  local variant=$1
  find "$outbase" -maxdepth 1 -type d -name "${variant}_*" | sort | tail -n 1 | xargs -I{} echo {}/model_best.pth.tar
}

train_and_eval() {
  local gpu=$1 variant=$2 tri=$3 aux=$4 weight=$5 mode=$6
  echo "[gpu$gpu] TRAIN $variant"
  bash "$repo/run_exact_rumpl_depth_20260724.sh" "$gpu" "$variant" "$tri" "$aux" "$weight" \
    || { echo "[FAIL] train $variant"; return 1; }
  local ckpt
  ckpt=$(find_ckpt "$variant")
  if [ ! -f "$ckpt" ]; then
    echo "[FAIL] no checkpoint for $variant"; return 1
  fi
  for v in 2 3 4 5; do
    echo "[gpu$gpu] EVAL $variant v$v"
    bash "$repo/eval_exact_depth_single_20260724.sh" "$gpu" "${variant}_v${v}" "$ckpt" "$v" "$mode" \
      > "$root/depth_anchor_eval/${variant}_v${v}.log" 2>&1 \
      || echo "[FAIL] eval $variant v$v"
  done
}

mkdir -p "$root/depth_anchor_eval"

gpu0_chain() {
  train_and_eval 0 D1_tri_anchor_seed0_20260724 1 0 0.1 d1_anchor
  train_and_eval 0 D3_anchor_depthaux_w01_seed0_20260724 1 1 0.1 d3_combo
}

gpu1_chain() {
  train_and_eval 1 D2_ray_depth_aux_w01_seed0_20260724 0 1 0.1 d2_depthaux
  train_and_eval 1 D4_ray_depth_aux_w03_seed0_20260724 0 1 0.3 d2_depthaux
}

gpu0_chain > "$root/phase3_gpu0_20260724.log" 2>&1 &
pid0=$!
gpu1_chain > "$root/phase3_gpu1_20260724.log" 2>&1 &
pid1=$!
echo "gpu0 chain pid=$pid0 log=$root/phase3_gpu0_20260724.log"
echo "gpu1 chain pid=$pid1 log=$root/phase3_gpu1_20260724.log"
wait $pid0 $pid1
echo "PHASE3 ALL DONE $(date --iso-8601=seconds)"
