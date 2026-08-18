#!/usr/bin/env bash
# Train one frozen-H76 hypothesis scorer. Run pose/joint variants in parallel.
set -euo pipefail

variant=${1:?usage: $0 <pose|joint> <physical-gpu> [smoke-batches]}
gpu=${2:?usage: $0 <pose|joint> <physical-gpu> [smoke-batches]}
smoke=${3:-0}
if [[ "${variant}" != "pose" && "${variant}" != "joint" ]]; then
  echo "variant must be pose or joint" >&2
  exit 2
fi

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811
TRAIN=${ROOT}/train_hypotheses
suffix=""
epochs=10
if [[ "${smoke}" -gt 0 ]]; then suffix="_smoke${smoke}"; epochs=1; fi
OUT=${ROOT}/C0_C1_training/${variant}${suffix}
mkdir -p "${OUT}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"${PY}" -u "${AUDIT}/train_h76_hypothesis_utility_20260811.py" \
  --train-shards \
    "${TRAIN}/H76_train_all_subsets_shard0of2.npz" \
    "${TRAIN}/H76_train_all_subsets_shard1of2.npz" \
  --validation-cache "${ROOT}/H76_validation_all_subsets.npz" \
  --variant "${variant}" --output-dir "${OUT}" \
  --epochs "${epochs}" --batch-size 512 --lr 5e-4 --weight-decay 1e-4 \
  --temperature 1.8 --workers 2 --seed 0 --gpu 0 \
  --smoke-batches "${smoke}" >"${OUT}/train.log" 2>&1

echo "completed ${variant}: ${OUT}"
