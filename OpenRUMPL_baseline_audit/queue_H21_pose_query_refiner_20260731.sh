#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {balanced|v2focus} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
TRAIN_HEATMAPS=${ROOT}/A0_h36m_train_heatmap_topk8
VAL_HEATMAPS=${ROOT}/A0_h36m_val_heatmap_topk8

case "${variant}" in
  balanced)
    probabilities=(1 1 1)
    output="${ROOT}/H21_pose_query_balanced"
    ;;
  v2focus)
    probabilities=(3 1 1)
    output="${ROOT}/H21_pose_query_v2focus"
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

for shard in $(seq 0 15); do
  while [[ ! -s "${TRAIN_HEATMAPS}/shard${shard}.npz" \
        || ! -s "${TRAIN_HEATMAPS}/shard${shard}.heatmaps.npy" ]]; do
    sleep 60
  done
done

# H21 follows the lower-cost H20/A0D/A1D screens.  This prevents either
# architecture from being judged under GPU contention.
while tmux has-session -t cjy_h20_random24 2>/dev/null \
   || tmux has-session -t cjy_h20_fixed2 2>/dev/null \
   || tmux has-session -t cjy_a0d_full_v2 2>/dev/null \
   || tmux has-session -t cjy_a0d_full_v34 2>/dev/null \
   || tmux has-session -t cjy_a1d_balanced 2>/dev/null \
   || tmux has-session -t cjy_a1d_v2focus 2>/dev/null; do
  sleep 60
done

mkdir -p "${output}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}"
if [[ ! -s "${output}/final.pth" ]]; then
  "${PY}" -u "${AUDIT}/train_iterative_pose_query_refiner.py" \
    --input-pkl \
      "${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --dense-shards "${TRAIN_HEATMAPS}"/shard{0..15}.npz \
    --steps 5000 \
    --view-probabilities "${probabilities[@]}" \
    --learning-rate 0.0003 \
    --device cuda:0 \
    --output-dir "${output}" \
    >"${output}/train.log" 2>&1
fi

if [[ ! -s "${output}/full_eval.json" ]]; then
  "${PY}" -u "${AUDIT}/eval_iterative_pose_query_refiner.py" \
    --input-pkl \
      "${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl" \
    --dense-shards "${VAL_HEATMAPS}"/shard{0..3}.npz \
    --checkpoint "${output}/final.pth" \
    --views 2 3 4 \
    --iterations 3 \
    --device cuda:0 \
    --output "${output}/full_eval.json" \
    >"${output}/full_eval.log" 2>&1
fi
