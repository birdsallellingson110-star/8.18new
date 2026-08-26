#!/usr/bin/env bash
# H15: export current GBT-aligned C2/H76 candidates on dense temporal H36M,
# append confidence-weighted candidates, and run a label-free-generation,
# label-only-evaluation temporal oracle audit. No training is performed here.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
TYPE=gbt_yolox_x_score001_fallback_legswap
PICKLE=${DATA}/data/datasets_mmpose/annot_temporal_5_5_${TYPE}/h36m_validation.pkl
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle
RAW=${OUT}/validation_c2_11c.npz
EXPANDED=${OUT}/validation_c2_22c.npz

test -s "${CFG}" && test -s "${CKPT}" && test -s "${PICKLE}"
mkdir -p "${OUT}"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0

if [[ ! -s "${RAW}" ]]; then
  cd "${REPO}"
  "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" \
    --dataset-name annot_temporal_5_5 \
    --mmpose-type "${TYPE}" --subset validation \
    --flip-lower-body-kp-test false --output "${RAW}" \
    --batch-size 256 --workers 8 --gpu 1 \
    >"${OUT}/export_h76.log" 2>&1
fi

if [[ ! -s "${EXPANDED}" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${RAW}" --output "${EXPANDED}" \
    --irls-iters 3 --huber-threshold-m 0.03 \
    >"${OUT}/append_confidence.log" 2>&1
fi

"${PY}" -u "${AUDIT}/audit_temporal_candidate_oracle_20260818.py" \
  --cache "${EXPANDED}" --validation-pkl "${PICKLE}" \
  --output-dir "${OUT}/audit" --window-lengths 1 3 5 9 \
  --frame-stride 5 --batch-size 512 \
  >"${OUT}/audit.log" 2>&1

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "H15 complete: ${OUT}"
