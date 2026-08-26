#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818
SCORE_DIR=${ROOT}/h16_temporal_c2_screen
TRAIN_CACHE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/train_c2_22c.npz
TRAIN_SCORES=${SCORE_DIR}/train_e2_scores.npy
TRAIN_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_CACHE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle/validation_c2_22c.npz
VAL_SCORES=${SCORE_DIR}/validation_e2_scores.npy
VAL_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl
TRAIN_FUSED=${ROOT}/h18_clean_temporal_cache/train/fused_poses.npy
VAL_FUSED=${ROOT}/h18_clean_temporal_cache/validation/fused_poses.npy
OUT=${ROOT}/h18_clean_temporal

export PYTHONPATH=${AUDIT}
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$(dirname "${TRAIN_FUSED}")" "$(dirname "${VAL_FUSED}")" "${OUT}"

if [[ ! -s "$(dirname "${TRAIN_FUSED}")/manifest.json" ]]; then
  "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${TRAIN_CACHE}" --scores "${TRAIN_SCORES}" \
    --output-dir "$(dirname "${TRAIN_FUSED}")" \
    --chunk-size 256 --gpu 0 \
    >"${OUT}/build_train.log" 2>&1
fi
if [[ ! -s "$(dirname "${VAL_FUSED}")/manifest.json" ]]; then
  "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${VAL_CACHE}" --scores "${VAL_SCORES}" \
    --output-dir "$(dirname "${VAL_FUSED}")" \
    --chunk-size 256 --gpu 0 \
    >"${OUT}/build_val.log" 2>&1
fi

if [[ ! -s "${OUT}/COMPLETED" ]]; then
  "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${TRAIN_FUSED}" \
    --train-pkl "${TRAIN_PKL}" \
    --validation-cache "${VAL_CACHE}" --validation-fused "${VAL_FUSED}" \
    --validation-pkl "${VAL_PKL}" --output-dir "${OUT}" \
    --window-length 9 --frame-stride 5 --epochs 8 --batch-size 64 \
    --hidden-dim 96 --layers 2 --lr 3e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed 0 \
    >"${OUT}/train.log" 2>&1
fi

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "H18 clean temporal complete: ${OUT}"
