#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TOPK=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/A0_h36m_train_heatmap_topk8
OUTPUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/A1_sparse_epipolar_transformer
LOG=${OUTPUT}/train.log

mkdir -p "${OUTPUT}"
while true; do
  complete=$(find "${TOPK}" -maxdepth 1 -name 'shard*.npz' -size +0c | wc -l)
  metadata=$(find "${TOPK}" -maxdepth 1 -name 'shard*.npz.json' -size +0c | wc -l)
  echo "$(date '+%F %T') waiting for A0 train top-K: ${complete}/16 data, ${metadata}/16 metadata"
  if [[ "${complete}" -eq 16 && "${metadata}" -eq 16 ]]; then
    break
  fi
  sleep 60
done

# The sparse branch is a bounded ablation after its oracle ceiling proved
# small.  Do not let it steal GPU time from H20 or the full dense diagnostic.
while tmux has-session -t cjy_h20_random24 2>/dev/null \
   || tmux has-session -t cjy_h20_fixed2 2>/dev/null \
   || tmux has-session -t cjy_a0d_full_v2 2>/dev/null \
   || tmux has-session -t cjy_a0d_full_v34 2>/dev/null; do
  sleep 60
done

if [[ -s "${OUTPUT}/history.json" ]]; then
  echo "A1 already complete: ${OUTPUT}/history.json"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
"${PY}" -u "${AUDIT}/train_sparse_epipolar_candidate_transformer.py" \
  --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl" \
  --topk-shards "${TOPK}"/shard{0..15}.npz \
  --output-dir "${OUTPUT}" \
  --device cuda:0 \
  --epochs 8 \
  --batch-groups 64 \
  --lr 0.001 \
  --dim 96 \
  --depth 2 \
  --heads 4 \
  >"${LOG}" 2>&1

echo "A1 training complete: ${OUTPUT}"
