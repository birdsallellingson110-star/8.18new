#!/usr/bin/env bash
# Zero-shot H36M-Occl evaluation of the frozen Stage-I ResNet spatial chain.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
FRONTEND=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/resnet_size_calibration/f015/h36m_validation_res152_occl.pkl
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full
HINGE=${BASE}/e2_c2_identity_hinge
OUT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/resnet_spatial
DATASET=annot_filtered_5_64_h36m_occl_f015_seed20260822_resnet
TYPE=res152_lt_alg_undistorted_annbox
TYPE_DIR=${DATA}/data/datasets_mmpose/${DATASET}_${TYPE}
GPU=${1:-0}

mkdir -p "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${FRONTEND}" && test -s "${BASE}/checkpoint.txt"
test -s "${HINGE}/seed0/model_best.pth.tar" && test -s "${HINGE}/seed1/model_best.pth.tar"
ln -sfn "${FRONTEND}" "${TYPE_DIR}/h36m_validation.pkl"

export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 RUMPL_TOKEN_DROPOUT=0

if [[ ! -s "${OUT}/validation_11c.npz" ]]; then
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "$(cat "${BASE}/checkpoint.txt")" \
    --dataset-name "${DATASET}" --mmpose-type "${TYPE}" --subset validation \
    --flip-lower-body-kp-test false --output "${OUT}/validation_11c.npz" \
    --batch-size 256 --workers 8 --gpu 0 >"${OUT}/export_11c.log" 2>&1
fi

"${PY}" -u "${AUDIT}/evaluate_h36m_occl_direct_cache_20260822.py" \
  --cache "${OUT}/validation_11c.npz" --output "${OUT}/direct_result.json" \
  >"${OUT}/direct_result.log" 2>&1

if [[ ! -s "${OUT}/validation_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${OUT}/validation_11c.npz" --output "${OUT}/validation_22c.npz" \
    >"${OUT}/append_22c.log" 2>&1
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
  --cache "${OUT}/validation_22c.npz" --checkpoint-root "${HINGE}" \
  --output "${OUT}/e2_identity_hinge_result.json" --v2-temperature 0.4 \
  --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
  >"${OUT}/e2_identity_hinge_result.log" 2>&1

sha256sum "${FRONTEND}" "${OUT}/validation_11c.npz" "${OUT}/validation_22c.npz" \
  "$(cat "${BASE}/checkpoint.txt")" "${HINGE}/seed0/model_best.pth.tar" \
  "${HINGE}/seed1/model_best.pth.tar" >"${OUT}/sha256.txt"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[H36M-Occl ResNet spatial] complete: ${OUT}"
