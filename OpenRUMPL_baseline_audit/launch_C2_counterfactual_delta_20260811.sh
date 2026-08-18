#!/usr/bin/env bash
set -euo pipefail
variant=${1:?usage: $0 <regression|balanced_rank> <physical-gpu> [seed]}
gpu=${2:?usage: $0 <regression|balanced_rank> <physical-gpu> [seed]}
seed=${3:-0}
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811
TRAIN=${ROOT}/train_hypotheses
OUT=${ROOT}/C2_delta_training/${variant}_seed${seed}
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"${PY}" -u "${AUDIT}/train_h76_counterfactual_delta_20260811.py" \
  --train-shards \
    "${TRAIN}/H76_train_all_subsets_shard0of2.npz" \
    "${TRAIN}/H76_train_all_subsets_shard1of2.npz" \
  --validation-cache "${ROOT}/H76_validation_all_subsets.npz" \
  --variant "${variant}" --output-dir "${OUT}" --epochs 10 \
  --batch-size 512 --lr 5e-4 --weight-decay 1e-4 --workers 2 \
  --seed "${seed}" --gpu 0 --selection-mode soft >"${OUT}/train.log" 2>&1
echo "completed ${variant}: ${OUT}"
