#!/usr/bin/env bash
set -euo pipefail

depth=${1:?usage: $0 <attention-depth:1|2> <physical-gpu> [smoke-batches] [seed]}
gpu=${2:?usage: $0 <attention-depth:1|2> <physical-gpu> [smoke-batches] [seed]}
smoke=${3:-0}
seed=${4:-0}
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
STAGE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811
suffix=""
if [[ "${smoke}" -gt 0 ]]; then suffix="_smoke${smoke}"; fi
if [[ "${seed}" -gt 0 ]]; then suffix="${suffix}_seed${seed}"; fi
OUT=${STAGE}/E_set_transformer_depth${depth}${suffix}
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=${AUDIT}

"${PY}" -u "${AUDIT}/train_h76_set_transformer_utility_20260811.py" \
  --train-shards \
  "${STAGE}/train_hypotheses/H76_train_all_subsets_shard0of2.npz" \
  "${STAGE}/train_hypotheses/H76_train_all_subsets_shard1of2.npz" \
  --validation-cache "${STAGE}/H76_validation_all_subsets.npz" \
  --output-dir "${OUT}" --attention-depth "${depth}" \
  --pretrain-epochs 10 --finetune-epochs 5 --batch-size 512 \
  --workers 4 --seed "${seed}" --gpu 0 --smoke-batches "${smoke}" \
  >"${OUT}/train.log" 2>&1

echo "finished E depth=${depth}: ${OUT}"
