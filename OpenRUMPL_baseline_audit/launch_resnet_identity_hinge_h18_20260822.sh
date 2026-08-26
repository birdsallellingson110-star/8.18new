#!/usr/bin/env bash
# Corrected ResNet temporal run: identity-hinge E2-C2 -> H18 T=9.
# This is deliberately separate from the earlier no-hinge H18 result.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
GPU=${1:-1}
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
CACHE=${BASE}/e2_c2_cache
HINGE=${BASE}/e2_c2_scorer_hinge/seed0/model_best.pth.tar
SCORE_DIR=${BASE}/e2_c2_hinge_temporal_scores
FUSED_DIR=${BASE}/h18_identity_hinge_fused
OUT=${BASE}/h18_identity_hinge
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
TEMPORAL_VALIDATION_PKL=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/frontend_temporal/validation/h36m_validation_res152_temporal.pkl
TRAIN_CACHE=${CACHE}/train_22c.npz
VAL_CACHE=${CACHE}/validation_temporal_22c.npz
TRAIN_SCORE=${SCORE_DIR}/train_e2_scores.npy
VAL_SCORE=${SCORE_DIR}/validation_temporal_e2_scores.npy

mkdir -p "${SCORE_DIR}" "${FUSED_DIR}" "${OUT}"
test -s "${HINGE}"
test -s "${TRAIN_CACHE}"
test -s "${VAL_CACHE}"
test -s "${FRONTEND}/train/h36m_train_res152.pkl"
test -s "${TEMPORAL_VALIDATION_PKL}"

export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

if [[ ! -s "${TRAIN_SCORE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${TRAIN_CACHE}" --checkpoint "${HINGE}" \
    --output "${TRAIN_SCORE}" --batch-size 256 --gpu 0 \
    >"${SCORE_DIR}/train.log" 2>&1
fi

if [[ ! -s "${VAL_SCORE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${VAL_CACHE}" --checkpoint "${HINGE}" \
    --output "${VAL_SCORE}" --batch-size 256 --gpu 0 \
    >"${SCORE_DIR}/validation_temporal.log" 2>&1
fi

if [[ ! -s "${FUSED_DIR}/train/manifest.json" ]]; then
  mkdir -p "${FUSED_DIR}/train"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${TRAIN_CACHE}" --scores "${TRAIN_SCORE}" \
    --output-dir "${FUSED_DIR}/train" --temperature-v2 0.4 \
    --temperature-v3 1.8 --temperature-v4 1.8 --chunk-size 256 --gpu 0 \
    >"${FUSED_DIR}/train.log" 2>&1
fi

if [[ ! -s "${FUSED_DIR}/validation/manifest.json" ]]; then
  mkdir -p "${FUSED_DIR}/validation"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${VAL_CACHE}" --scores "${VAL_SCORE}" \
    --output-dir "${FUSED_DIR}/validation" --temperature-v2 0.4 \
    --temperature-v3 1.8 --temperature-v4 1.8 --chunk-size 256 --gpu 0 \
    >"${FUSED_DIR}/validation.log" 2>&1
fi

if [[ ! -s "${OUT}/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u \
    "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" \
    --train-fused "${FUSED_DIR}/train/fused_poses.npy" \
    --train-pkl "${FRONTEND}/train/h36m_train_res152.pkl" \
    --validation-cache "${VAL_CACHE}" \
    --validation-fused "${FUSED_DIR}/validation/fused_poses.npy" \
    --validation-pkl "${TEMPORAL_VALIDATION_PKL}" \
    --output-dir "${OUT}" --window-length 9 --frame-stride 5 \
    --epochs 12 --batch-size 64 --hidden-dim 96 --layers 2 \
    --lr 5e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed 0 >"${OUT}/train.log" 2>&1
fi

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet identity-hinge H18] complete ${OUT}"
