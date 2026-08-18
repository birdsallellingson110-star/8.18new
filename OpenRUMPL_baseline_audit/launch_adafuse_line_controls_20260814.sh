#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {v2|v34} PHYSICAL_GPU" >&2
  exit 2
fi
part=$1
gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/AdaFuseLineControls_20260814
VAL=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/A0_h36m_val_heatmap_topk8

if [[ "${part}" == v2 ]]; then
  views=(2)
else
  views=(3 4)
fi
mkdir -p "${ROOT}/${part}"
export CUDA_VISIBLE_DEVICES="${gpu}"
"${PY}" -u "${AUDIT}/eval_h36m_dense_epipolar_heatmaps.py" \
  --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl" \
  --dense-shards "${VAL}"/shard{0..3}.npz \
  --views "${views[@]}" \
  --alphas 0.25 0.5 1.0 \
  --depth-min-m 1.0 \
  --depth-max-m 5.0 \
  --depth-samples 2 \
  --support-mode line \
  --device cuda:0 \
  --output "${ROOT}/${part}/full_eval.json" \
  >"${ROOT}/${part}/full_eval.log" 2>&1
