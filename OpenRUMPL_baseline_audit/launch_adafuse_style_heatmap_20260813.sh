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
    output="${ROOT}/AdaFuseStyle_balanced_20260813"
    ;;
  v2focus)
    probabilities=(3 1 1)
    output="${ROOT}/AdaFuseStyle_v2focus_20260813"
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
mkdir -p "${output}"

if [[ ! -s "${output}/final.pth" ]]; then
  "${PY}" -u "${AUDIT}/train_dense_geometry_residual_fusion.py" \
    --model-kind adafuse_style \
    --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --dense-shards "${TRAIN_HEATMAPS}"/shard{0..15}.npz \
    --steps 6000 \
    --depth-samples 32 \
    --view-probabilities "${probabilities[@]}" \
    --learning-rate 0.0003 \
    --device cuda:0 \
    --output-dir "${output}" \
    >"${output}/train.log" 2>&1
fi

if [[ ! -s "${output}/full_eval.json" ]]; then
  "${PY}" -u "${AUDIT}/eval_h36m_dense_epipolar_heatmaps.py" \
    --input-pkl "${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl" \
    --dense-shards "${VAL_HEATMAPS}"/shard{0..3}.npz \
    --views 2 3 4 \
    --alphas 0.25 \
    --depth-samples 32 \
    --fusion-checkpoint "${output}/final.pth" \
    --fusion-model-kind adafuse_style \
    --device cuda:0 \
    --output "${output}/full_eval.json" \
    >"${output}/full_eval.log" 2>&1
fi
