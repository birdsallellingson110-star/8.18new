#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: $0 <view_only|hierarchical> <physical-gpu> [smoke-batches]}
gpu=${2:?usage: $0 <view_only|hierarchical> <physical-gpu> [smoke-batches]}
smoke=${3:-0}
case "${variant}" in
  view_only) depth=0 ;;
  hierarchical) depth=2 ;;
  *) echo "variant must be view_only or hierarchical" >&2; exit 2 ;;
esac
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
STAGE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811
suffix=""
if [[ "${smoke}" -gt 0 ]]; then suffix="_smoke${smoke}"; fi
OUT=${STAGE}/F_${variant}_ray_view_attention${suffix}
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=${AUDIT}

"${PY}" -u "${AUDIT}/train_h76_set_transformer_utility_20260811.py" \
  --train-shards \
  "${STAGE}/train_hypotheses/H76_train_all_subsets_shard0of2.npz" \
  "${STAGE}/train_hypotheses/H76_train_all_subsets_shard1of2.npz" \
  --validation-cache "${STAGE}/H76_validation_all_subsets.npz" \
  --output-dir "${OUT}" --attention-depth "${depth}" --view-cross-attention \
  --pretrain-epochs 10 --finetune-epochs 5 --batch-size 512 \
  --workers 4 --seed 0 --gpu 0 --smoke-batches "${smoke}" \
  >"${OUT}/train.log" 2>&1

echo "finished F ${variant}: ${OUT}"
