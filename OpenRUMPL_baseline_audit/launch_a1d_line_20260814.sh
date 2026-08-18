#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {balanced|v2focus} PHYSICAL_GPU" >&2
  exit 2
fi
variant=$1
gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
TRAIN=${ROOT}/A0_h36m_train_heatmap_topk8
VAL=${ROOT}/A0_h36m_val_heatmap_topk8
case "${variant}" in
  balanced) probs=(1 1 1); out=${ROOT}/A1D_line_balanced_20260814 ;;
  v2focus) probs=(3 1 1); out=${ROOT}/A1D_line_v2focus_20260814 ;;
  *) echo "unknown variant" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${gpu}"
mkdir -p "${out}"
if [[ ! -s "${out}/final.pth" ]]; then
  "${PY}" -u "${AUDIT}/train_dense_geometry_residual_fusion.py" \
    --model-kind a1d --support-mode line \
    --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --dense-shards "${TRAIN}"/shard{0..15}.npz \
    --steps 6000 --depth-min-m 1 --depth-max-m 5 --depth-samples 2 \
    --view-probabilities "${probs[@]}" --learning-rate 0.0003 \
    --device cuda:0 --output-dir "${out}" \
    >"${out}/train.log" 2>&1
fi
if [[ ! -s "${out}/full_eval.json" ]]; then
  "${PY}" -u "${AUDIT}/eval_h36m_dense_epipolar_heatmaps.py" \
    --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl" \
    --dense-shards "${VAL}"/shard{0..3}.npz \
    --views 2 3 4 --alphas 0.25 --depth-min-m 1 --depth-max-m 5 \
    --depth-samples 2 --support-mode line --fusion-checkpoint "${out}/final.pth" \
    --fusion-model-kind a1d --device cuda:0 --output "${out}/full_eval.json" \
    >"${out}/full_eval.log" 2>&1
fi
