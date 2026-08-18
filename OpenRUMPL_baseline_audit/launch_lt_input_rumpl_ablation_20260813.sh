#!/usr/bin/env bash
# Strict same-input ablation: public RUMPL vs H76 geometry representation.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {R0|TA|H76} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_input_rumpl_ablation
TYPE=lt_alg_undistorted_annbox
DATA_DIR=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${TYPE}

case "${variant}" in
  R0)
    tri_anchor=0
    centered=0
    plucker=0
    description=originalRUMPL
    ;;
  TA)
    tri_anchor=1
    centered=0
    plucker=0
    description=triAnchorOnly
    ;;
  H76)
    tri_anchor=1
    centered=1
    plucker=1
    description=triAnchor_centeredPlucker
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

tag="LTIN2_${variant}_${description}_sameProtocol_noFlip_seed0_20260813"
model_dir=$(find "${MODEL_ROOT}" -maxdepth 1 -type d -name "${tag}_*" -print | sort | tail -n 1)
mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/eval/${variant}"
test -s "${CFG}"
test -s "${DATA_DIR}/h36m_train.pkl"
test -s "${DATA_DIR}/h36m_validation.pkl"

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_TRI_ANCHOR="${tri_anchor}" RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS="${centered}" RUMPL_INPUT_PLUCKER="${plucker}"
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0
export MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all

cd "${REPO}"
if [[ -z "${model_dir}" || ! -s "${model_dir}/model_best.pth.tar" ]]; then
  log=${BASE}/logs/${tag}.log
  {
    echo "[LTIN] start variant=${variant} gpu=${gpu} $(date --iso-8601=seconds)"
    echo "[LTIN] only_variable tri_anchor=${tri_anchor} centered=${centered} plucker=${plucker}"
    sha256sum "${CFG}" "${DATA_DIR}/h36m_train.pkl" "${DATA_DIR}/h36m_validation.pkl"
  } | tee "${log}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 12 --seed 0 \
    --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
    --validate-on-two-datasets 0 --use-mmpose-val 1 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
    >> "${log}" 2>&1
  model_dir=$(find "${MODEL_ROOT}" -maxdepth 1 -type d -name "${tag}_*" -print | sort | tail -n 1)
fi

checkpoint=${model_dir}/model_best.pth.tar
test -s "${checkpoint}"
printf '%s\n' "${checkpoint}" > "${BASE}/checkpoints/${variant}.txt"

for n_views in 2 3 4; do
  eval_dir=${BASE}/eval/${variant}/V${n_views}
  mkdir -p "${eval_dir}"
  RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
    --test-on-all-cameras true --n-views-combinations "${n_views}" \
    --model-num-views 4 --test-mmpose-type "${TYPE}" \
    > "${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" > "${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds > "${BASE}/${variant}.done"
