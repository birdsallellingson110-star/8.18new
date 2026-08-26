#!/usr/bin/env bash
# H17: pairwise V2 candidate utility residual; no frontend or ResNet change.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h17_v2_pairwise_utility

TRAIN_CACHE=${BASE}/train_c2_22c.npz
TRAIN_SCORES=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h16_temporal_c2_screen/train_e2_scores.npy
VAL_CACHE=${VAL}/validation_c2_22c.npz
VAL_SCORES=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h16_temporal_c2_screen/validation_e2_scores.npy

mkdir -p "${OUT}"
test -s "${TRAIN_CACHE}" && test -s "${TRAIN_SCORES}" \
  && test -s "${VAL_CACHE}" && test -s "${VAL_SCORES}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1

if [[ ! -s "${OUT}/result.json" ]]; then
  "${PY}" -u "${AUDIT}/train_h17_v2_pairwise_utility_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-scores "${TRAIN_SCORES}" \
    --validation-cache "${VAL_CACHE}" --validation-scores "${VAL_SCORES}" \
    --output-dir "${OUT}" --epochs 15 --batch-size 1024 \
    --lr 5e-4 --ght-weight 0.1 --identity-weight 0.5 \
    --seed 0 --gpu 0 >"${OUT}/train.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "H17 complete: ${OUT}"
