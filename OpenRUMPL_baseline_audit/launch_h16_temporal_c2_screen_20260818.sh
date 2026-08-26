#!/usr/bin/env bash
# H16: short temporal utility screen on the frozen calibrated E2-C2 line.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h16_temporal_c2_screen
CKPT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2/seed0/model_best.pth.tar
TRAIN_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_temporal_5_5_gbt_yolox_x_score001_fallback_legswap/h36m_validation.pkl
TRAIN_CACHE=${BASE}/train_c2_22c.npz
VAL_CACHE=${VAL}/validation_c2_22c.npz
TRAIN_SCORES=${OUT}/train_e2_scores.npy
VAL_SCORES=${OUT}/validation_e2_scores.npy

mkdir -p "${OUT}"
test -s "${TRAIN_CACHE}" && test -s "${VAL_CACHE}" && test -s "${CKPT}" \
  && test -s "${TRAIN_PKL}" && test -s "${VAL_PKL}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1

if [[ ! -s "${TRAIN_SCORES}" || ! -s "${TRAIN_SCORES%.npy}.json" ]]; then
  "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${TRAIN_CACHE}" --checkpoint "${CKPT}" \
    --output "${TRAIN_SCORES}" --batch-size 256 --gpu 0 \
    >"${OUT}/build_train_scores.log" 2>&1
fi
if [[ ! -s "${VAL_SCORES}" || ! -s "${VAL_SCORES%.npy}.json" ]]; then
  "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${VAL_CACHE}" --checkpoint "${CKPT}" \
    --output "${VAL_SCORES}" --batch-size 256 --gpu 0 \
    >"${OUT}/build_validation_scores.log" 2>&1
fi

if [[ ! -s "${OUT}/result.json" ]]; then
  "${PY}" -u "${AUDIT}/train_temporal_e2_c2_screen_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-pkl "${TRAIN_PKL}" \
    --train-scores "${TRAIN_SCORES}" \
    --validation-cache "${VAL_CACHE}" --validation-pkl "${VAL_PKL}" \
    --validation-scores "${VAL_SCORES}" --output-dir "${OUT}" \
    --window-length 9 --epochs 4 --direct-epochs 2 \
    --batch-size 128 --lr 3e-4 --ght-lr 5e-5 \
    --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
    --identity-weight 0.5 --workers 0 --seed 0 --max-train-windows 4096 \
    --max-holdout-windows 2048 --gpu 0 \
    >"${OUT}/train.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "H16 complete: ${OUT}"
