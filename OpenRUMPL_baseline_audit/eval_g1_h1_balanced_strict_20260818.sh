#!/usr/bin/env bash
# Evaluation-only recovery for G1. The 20-epoch training completed, but the
# original launcher was edited while its shell was still alive and its
# post-training evaluation command was parsed incorrectly. Do not retrain.
set -euo pipefail

PY=${PY:-/home/lixiaob/cjy/rumpl_venv310/bin/python}
AUDIT=${AUDIT:-/home/lixiaob/cjy/OpenRUMPL_baseline_audit}
REPO=${REPO:-/home/lixiaob/cjy/OpenRUMPL/RUMPL}
CFG=${CFG:-/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml}
TYPE=${TYPE:-gbt_yolox_x_score001_fallback_legswap}
CKPT=${CKPT:-/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/G1_H1_BALANCED_MIXED_T1_20E_LR1E5_seed0_20260818_2026-08-18_01-01-28/model_best.pth.tar}
OUT=${OUT:-/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h1_balanced_continuation}

test -s "${CFG}"
test -s "${CKPT}"
mkdir -p "${OUT}"
printf '%s\n' "${CKPT}" >"${OUT}/checkpoint.txt"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
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

cd "${REPO}"
for views in 2 3 4; do
  eval_dir="${OUT}/eval/V${views}"
  mkdir -p "${eval_dir}"
  RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${CKPT}" --output-dir "${eval_dir}" \
    --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
    --test-on-all-cameras true --n-views-combinations "${views}" \
    --model-num-views 4 --test-mmpose-type "${TYPE}" \
    >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  test -s "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "G1 strict evaluation complete $(date --iso-8601=seconds)"
