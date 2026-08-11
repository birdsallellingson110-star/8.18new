#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
TOPK=${ROOT}/A0_h36m_val_heatmap_topk8
TRAIN=${ROOT}/A1_sparse_epipolar_transformer
RESULT=${ROOT}/A1_sparse_epipolar_transformer_full_eval.json
LOG=${ROOT}/A1_sparse_epipolar_transformer_full_eval.log

while [[ ! -s "${TRAIN}/history.json" || ! -s "${TRAIN}/checkpoint_best.pth" ]]; do
  echo "$(date '+%F %T') waiting for completed A1 training"
  sleep 60
done

if [[ -s "${RESULT}" ]]; then
  echo "A1 evaluation already complete: ${RESULT}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
"${PY}" -u "${AUDIT}/eval_h36m_sparse_epipolar_topk.py" \
  --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl" \
  --topk-shards \
    "${TOPK}/shard0.npz" \
    "${TOPK}/shard1.npz" \
    "${TOPK}/shard2.npz" \
    "${TOPK}/shard3.npz" \
  --views 2 3 4 \
  --skip-epipolar \
  --candidate-transformer-checkpoint "${TRAIN}/checkpoint_best.pth" \
  --device cuda:0 \
  --output "${RESULT}" \
  >"${LOG}" 2>&1

echo "A1 formal evaluation complete: ${RESULT}"
