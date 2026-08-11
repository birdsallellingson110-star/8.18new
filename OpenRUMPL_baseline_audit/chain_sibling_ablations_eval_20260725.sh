#!/usr/bin/env bash
# Evaluate sibling ablations (already trained) on V2–V5 × occ{0,0.3,0.6}.
# usage: $0 GPU TAG_PREFIX VARIANT MODE
#   MODE: adafuse_vw | baseline
set -uo pipefail
GPU=${1:?}
TAG_PREFIX=${2:?}
VARIANT=${3:?}
MODE=${4:?}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
mkdir -p "$root"

ckpt=$(find "$outbase" -maxdepth 1 -type d -name "${VARIANT}_*" | sort | tail -1)/model_best.pth.tar
[ -f "$ckpt" ] || { echo "[FAIL] no ckpt for $VARIANT"; exit 1; }
echo "[${TAG_PREFIX}] gpu=$GPU mode=$MODE ckpt=$ckpt"

# C was trained with 2D refine; keep OFF at eval (train-time aug only), matching S2a/A2.
export RUMPL_2D_REFINE=0 RUMPL_OCC_JOINT_LOSS=0 RUMPL_TRAIN_STRUCT_OCC=0

for v in 2 3 4 5; do
  for occ in 0.0 0.3 0.6; do
    tag="${TAG_PREFIX}_v${v}_occ${occ}"
    if [ -f "$root/${tag}_summary.json" ]; then
      echo "[skip] $tag"; continue
    fi
    echo "[gpu${GPU}] $tag $(date +%H:%M:%S)"
    bash "$repo/eval_occlusion_single_20260724.sh" "$GPU" "$tag" "$ckpt" "$v" "$MODE" "$occ" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag"
  done
done
echo "[${TAG_PREFIX}] done $(date -Iseconds)"
