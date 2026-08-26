#!/usr/bin/env bash
# Geometry-uncertainty H76 -> E2-C2 -> H18 temporal continuation.
# The E2 scorer is frame-level; this script adds the dense temporal export,
# frame fusion cache, and the actual T=9 residual trainer after calibration.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
ROOT=${BASE}/h76_geom_uncertainty
E2=${BASE}/e2_c2_geom_uncertainty
SCORER=${E2}/scorer
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
TEMP_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/frontend_temporal
GPU=${1:-1}

TYPE=res152_lt_alg_undistorted_annbox
TEMP_DATASET=annot_filtered_5_64_res152_lt_alg_undistorted_annbox_geom_temporal
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/${TEMP_DATASET}_${TYPE}
TEMP_CACHE=${E2}/temporal_cache
SCORE_DIR=${BASE}/e2_c2_geom_temporal_scores
FUSED_DIR=${BASE}/h18_geom_e2_fused
OUT=${BASE}/h18_geom_e2

mkdir -p "${TYPE_DIR}" "${TEMP_CACHE}" "${SCORE_DIR}" "${FUSED_DIR}" "${OUT}"
ln -sfn "${FRONTEND}/train/h36m_train_res152.pkl" "${TYPE_DIR}/h36m_train.pkl"
ln -sfn "${TEMP_FRONTEND}/validation/h36m_validation_res152_temporal.pkl" "${TYPE_DIR}/h36m_validation.pkl"

export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=1
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 RUMPL_TOKEN_DROPOUT=0

# Export the dense temporal validation pool using the completed H76 checkpoint.
if [[ ! -s "${TEMP_CACHE}/validation_11c.npz" ]]; then
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "$(cat "${ROOT}/seed0/checkpoint.txt")" \
    --dataset-name "${TEMP_DATASET}" --mmpose-type "${TYPE}" --subset validation \
    --flip-lower-body-kp-test false --output "${TEMP_CACHE}/validation_11c.npz" \
    --batch-size 256 --workers 8 --gpu 0 >"${TEMP_CACHE}/validation_11c.log" 2>&1
fi
if [[ ! -s "${TEMP_CACHE}/validation_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${TEMP_CACHE}/validation_11c.npz" --output "${TEMP_CACHE}/validation_22c.npz" \
    >"${TEMP_CACHE}/append_validation.log" 2>&1
fi

# The scorer is frame-level, but it is the calibrated scorer used to build the
# temporal fused pose. Wait for both seeds and calibration without rerunning it.
while [[ ! -s "${SCORER}/calibrated_v2t04.json" ]]; do sleep 30; done
HINGE="${SCORER}/seed0/model_best.pth.tar"
TRAIN_CACHE="${E2}/cache/train_22c.npz"
VAL_CACHE="${TEMP_CACHE}/validation_22c.npz"
TRAIN_SCORE="${SCORE_DIR}/train_e2_scores.npy"
VAL_SCORE="${SCORE_DIR}/validation_temporal_e2_scores.npy"
test -s "${TRAIN_CACHE}"; test -s "${VAL_CACHE}"

if [[ ! -s "${TRAIN_SCORE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${TRAIN_CACHE}" --checkpoint "${HINGE}" --output "${TRAIN_SCORE}" \
    --batch-size 256 --gpu 0 >"${SCORE_DIR}/train.log" 2>&1
fi
if [[ ! -s "${VAL_SCORE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${VAL_CACHE}" --checkpoint "${HINGE}" --output "${VAL_SCORE}" \
    --batch-size 256 --gpu 0 >"${SCORE_DIR}/validation_temporal.log" 2>&1
fi

if [[ ! -s "${FUSED_DIR}/train/manifest.json" ]]; then
  mkdir -p "${FUSED_DIR}/train"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${TRAIN_CACHE}" --scores "${TRAIN_SCORE}" --output-dir "${FUSED_DIR}/train" \
    --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
    --chunk-size 256 --gpu 0 >"${FUSED_DIR}/train.log" 2>&1
fi
if [[ ! -s "${FUSED_DIR}/validation/manifest.json" ]]; then
  mkdir -p "${FUSED_DIR}/validation"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${VAL_CACHE}" --scores "${VAL_SCORE}" --output-dir "${FUSED_DIR}/validation" \
    --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
    --chunk-size 256 --gpu 0 >"${FUSED_DIR}/validation.log" 2>&1
fi

if [[ ! -s "${OUT}/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${FUSED_DIR}/train/fused_poses.npy" \
    --train-pkl "${FRONTEND}/train/h36m_train_res152.pkl" \
    --validation-cache "${VAL_CACHE}" --validation-fused "${FUSED_DIR}/validation/fused_poses.npy" \
    --validation-pkl "${TEMP_FRONTEND}/validation/h36m_validation_res152_temporal.pkl" \
    --output-dir "${OUT}" --window-length 9 --frame-stride 5 --epochs 12 \
    --batch-size 64 --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 \
    --residual-scale-m 0.10 --gpu 0 --seed 0 >"${OUT}/train.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet geometry-E2 H18] complete ${OUT}"
