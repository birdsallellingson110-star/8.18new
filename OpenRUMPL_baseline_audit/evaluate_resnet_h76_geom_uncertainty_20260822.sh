#!/usr/bin/env bash
# Evaluate the geometry-uncertainty H76 pair under the same strict protocol.
set -euo pipefail
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=res152_lt_alg_undistorted_annbox
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
ROOT=${BASE}/h76_geom_uncertainty
OUT=${ROOT}/eval
mkdir -p "${OUT}"
eval_seed() {
  local seed="$1" gpu="$2" ckpt="$3"
  local out="${OUT}/seed${seed}"
  mkdir -p "${out}"
  for views in 2 3 4; do
    local eval_dir="${out}/V${views}"
    [[ -s "${eval_dir}/COMPLETED" ]] && continue
    mkdir -p "${eval_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${AUDIT}" RUMPL_EVAL_STRICT=1 \
      RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05 \
      RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1 \
      RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0 \
      RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=1 \
      "${PY}" -u "${REPO}/run/eval_rumpl_checkpoint.py" \
      --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
      --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
      --test-on-all-cameras true --n-views-combinations "${views}" \
      --model-num-views 4 --test-mmpose-type "${TYPE}" \
      >"${eval_dir}/eval.log" 2>&1
    prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    test -s "${prediction}"
    "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${prediction}" \
      --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
    date --iso-8601=seconds >"${eval_dir}/COMPLETED"
  done
}
seed0="${ROOT}/seed0/checkpoint.txt"; seed1="${ROOT}/seed1/checkpoint.txt"
test -s "${seed0}" && test -s "${seed1}"
eval_seed 0 0 "$(cat "${seed0}")" & p0=$!
eval_seed 1 1 "$(cat "${seed1}")" & p1=$!
wait "${p0}" "${p1}"
echo "[ResNet-H76-geom-eval] complete $(date --iso-8601=seconds)"
