#!/usr/bin/env bash
# Continue the ResNet E2-C2 -> H18-lowLR line after replacing the sparse
# ordinary validation frontend with the dense (step-5) official LT frontend.
# This is necessary for a causal T=9, stride=5 validation protocol.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE=res152_lt_alg_undistorted_annbox
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_temporal_5_5_${TYPE}
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
TEMP_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/frontend_temporal/validation
TEMP_PKL=${TEMP_FRONTEND}/h36m_validation_res152_temporal.pkl
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
CKPT=${BASE}/h76/checkpoint.txt
CACHE=${BASE}/e2_c2_cache
SCORER=${BASE}/e2_c2_scorer
SCORES=${BASE}/h18_lowlr_fused_dense_validation_scores
FUSED=${BASE}/h18_lowlr_fused_dense_validation
OUT=${BASE}/h18_lowlr_dense_validation

mkdir -p "${TYPE_DIR}" "${SCORES}" "${FUSED}" "${OUT}"
test -s "${TEMP_PKL}" && test -s "${CKPT}" && test -s "${SCORER}/seed0/model_best.pth.tar"

link="${TYPE_DIR}/h36m_validation.pkl"
if [[ -e "${link}" || -L "${link}" ]]; then
  [[ "$(readlink -f "${link}")" == "$(readlink -f "${TEMP_PKL}")" ]] || {
    echo "mismatched temporal validation link: ${link}" >&2; exit 2;
  }
else
  ln -s "${TEMP_PKL}" "${link}"
fi

export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0

VAL_CACHE=${CACHE}/validation_temporal_11c.npz
VAL_CACHE22=${CACHE}/validation_temporal_22c.npz
if [[ ! -s "${VAL_CACHE}" ]]; then
  CUDA_VISIBLE_DEVICES=0 cd "${REPO}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "$(cat "${CKPT}")" \
    --dataset-name annot_temporal_5_5 --mmpose-type "${TYPE}" --subset validation \
    --flip-lower-body-kp-test false --output "${VAL_CACHE}" \
    --batch-size 256 --workers 8 --gpu 0 \
    >"${CACHE}/validation_temporal_export.log" 2>&1
fi
if [[ ! -s "${VAL_CACHE22}" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${VAL_CACHE}" --output "${VAL_CACHE22}" \
    >"${CACHE}/append_validation_temporal.log" 2>&1
fi

VAL_SCORE=${SCORES}/validation_temporal_e2_scores.npy
if [[ ! -s "${VAL_SCORE}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${VAL_CACHE22}" --checkpoint "${SCORER}/seed0/model_best.pth.tar" \
    --output "${VAL_SCORE}" --batch-size 256 --gpu 0 \
    >"${SCORES}/validation_temporal.log" 2>&1
fi
if [[ ! -s "${FUSED}/validation/manifest.json" ]]; then
  mkdir -p "${FUSED}/validation"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${VAL_CACHE22}" --scores "${VAL_SCORE}" \
    --output-dir "${FUSED}/validation" --chunk-size 256 --gpu 0 \
    >"${FUSED}/validation.log" 2>&1
fi

TRAIN_CACHE=${CACHE}/train_22c.npz
TRAIN_FUSED=${BASE}/h18_lowlr_fused/train/fused_poses.npy
TRAIN_PKL=${FRONTEND}/train/h36m_train_res152.pkl
test -s "${TRAIN_CACHE}" && test -s "${TRAIN_FUSED}" && test -s "${TRAIN_PKL}"
if [[ ! -s "${OUT}/result.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${TRAIN_FUSED}" --train-pkl "${TRAIN_PKL}" \
    --validation-cache "${VAL_CACHE22}" --validation-fused "${FUSED}/validation/fused_poses.npy" \
    --validation-pkl "${TEMP_PKL}" --output-dir "${OUT}" \
    --window-length 9 --frame-stride 5 --epochs 12 --batch-size 64 \
    --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed 0 >"${OUT}/train.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-H18 dense-validation] complete: ${OUT}"
