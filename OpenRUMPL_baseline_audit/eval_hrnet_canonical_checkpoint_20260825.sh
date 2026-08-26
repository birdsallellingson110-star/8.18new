#!/usr/bin/env bash
set -euo pipefail
: "${1:?checkpoint path required}"
: "${2:?output root required}"

CKPT=$1
ROOT=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
test -s "${CKPT}"
mkdir -p "${ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_BODY_CANONICAL_FRAME=1 RUMPL_BODY_CANONICAL_REG=1e-4
export RUMPL_BODY_CANONICAL_PELVIS_PRIOR=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
for views in 2 3 4; do
  out="${ROOT}/V${views}"
  mkdir -p "${out}"
  RUMPL_EVAL_STRICT=1 "${PY}" -u "${REPO}/run/eval_rumpl_checkpoint.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" --output-dir "${out}" \
    --workers 8 --gpu 0 --use-mmpose-val true \
    --flip-lower-body-kp-test false --test-on-all-cameras true \
    --n-views-combinations "${views}" --model-num-views 4 \
    --test-mmpose-type "${TYPE}" >"${out}/eval.log" 2>&1
  pred=$(find "${out}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
    --output-json "${out}/table2.json" >"${out}/table2.log" 2>&1
done
date --iso-8601=seconds >"${ROOT}/COMPLETED"
