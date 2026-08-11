#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TOPK=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/A0_h36m_val_heatmap_topk8
OUTPUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
RESULT=${OUTPUT}/A0S_sparse_epipolar_full_grid.json
LOG=${OUTPUT}/A0S_sparse_epipolar_full_grid.log

if [[ -s "${RESULT}" ]]; then
  echo "already complete: ${RESULT}"
  exit 0
fi

"${PY}" -u "${AUDIT}/eval_h36m_sparse_epipolar_topk.py" \
  --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl" \
  --topk-shards \
    "${TOPK}/shard0.npz" \
    "${TOPK}/shard1.npz" \
    "${TOPK}/shard2.npz" \
    "${TOPK}/shard3.npz" \
  --views 2 3 4 \
  --sigma-m 0.05 0.1 \
  --unary-weight 2 4 \
  --output "${RESULT}" \
  >"${LOG}" 2>&1

echo "complete: ${RESULT}"
