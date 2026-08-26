#!/usr/bin/env bash
# Zero-shot H36M-Occl evaluation of the frozen Stage-I HRNet spatial chain.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
FRONTEND_ROOT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/hrnet_frontend
FRONTEND=${FRONTEND_ROOT}/merged/h36m_validation_hrnet_occl.pkl
CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
SCORER=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2
OUT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/hrnet_spatial
DATASET=annot_filtered_5_64
TYPE=gbt_h36m_occl_f015_seed20260822_hrnet
TYPE_DIR=${DATA}/data/datasets_mmpose/${DATASET}_${TYPE}
GPU=${1:-1}

mkdir -p "${OUT}" "${TYPE_DIR}"
while [[ ! -s "${FRONTEND_ROOT}/COMPLETED" ]]; do sleep 10; done
test -s "${CFG}" && test -s "${FRONTEND}" && test -s "${CKPT}"
test -s "${SCORER}/seed0/model_best.pth.tar" && test -s "${SCORER}/seed1/model_best.pth.tar"
ln -sfn "${FRONTEND}" "${TYPE_DIR}/h36m_validation.pkl"

export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0 GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0

if [[ ! -s "${OUT}/validation_11c.npz" ]]; then
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" --dataset-name "${DATASET}" \
    --mmpose-type "${TYPE}" --subset validation --flip-lower-body-kp-test false \
    --output "${OUT}/validation_11c.npz" --batch-size 256 --workers 8 --gpu 0 \
    >"${OUT}/export_11c.log" 2>&1
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
  --cache "${OUT}/validation_22c.npz" --checkpoint-root "${SCORER}" \
  --output "${OUT}/e2_c2_result.json" --v2-temperature 0.4 \
  --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
  >"${OUT}/e2_c2_result.log" 2>&1

sha256sum "${FRONTEND}" "${OUT}/validation_11c.npz" "${OUT}/validation_22c.npz" \
  "${CKPT}" "${SCORER}/seed0/model_best.pth.tar" \
  "${SCORER}/seed1/model_best.pth.tar" >"${OUT}/sha256.txt"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[H36M-Occl HRNet spatial] complete: ${OUT}"
