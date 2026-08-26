#!/usr/bin/env bash
# Generic isolated HRNet canonical-repair branch. Required variables:
# REPAIR_NAME, REPAIR_LR, REPAIR_EPOCHS, REPAIR_LR_STEPS,
# REPAIR_PELVIS_PRIOR and REPAIR_BODY_REG.
set -euo pipefail

: "${REPAIR_NAME:?set REPAIR_NAME}"
: "${REPAIR_LR:?set REPAIR_LR}"
: "${REPAIR_EPOCHS:?set REPAIR_EPOCHS}"
: "${REPAIR_LR_STEPS:?set REPAIR_LR_STEPS}"
: "${REPAIR_PELVIS_PRIOR:?set REPAIR_PELVIS_PRIOR to 0 or 1}"
: "${REPAIR_BODY_REG:?set REPAIR_BODY_REG}"

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
PHASE1=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/phase1_cont20_lr1e5
DEFAULT_INIT=$(cat "${PHASE1}/final_checkpoint.txt")
INIT=${REPAIR_INIT:-${DEFAULT_INIT}}
VISIBLE_GPU=${REPAIR_VISIBLE_GPU:-0}
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/branches_20260825/${REPAIR_NAME}
TAG=CAMGEN_HRNET_CANON_REPAIR_${REPAIR_NAME}_seed0_20260825

TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl

mkdir -p "${ROOT}"
test -s "${CFG}"; test -s "${INIT}"
test -s "${TRAIN_PKL}"; test -s "${VAL_PKL}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_END_EPOCH="${REPAIR_EPOCHS}"
export RUMPL_INIT_CHECKPOINT="${INIT}"
export RUMPL_FINETUNE_LR="${REPAIR_LR}"
export RUMPL_LR_STEPS="${REPAIR_LR_STEPS}"
export RUMPL_SAVE_EVERY_N_EPOCHS=1
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_BODY_CANONICAL_FRAME=1
export RUMPL_BODY_CANONICAL_REG="${REPAIR_BODY_REG}"
export RUMPL_BODY_CANONICAL_PELVIS_PRIOR="${REPAIR_PELVIS_PRIOR}"
export RUMPL_BODY_CANONICAL_ROBUST_TORSO="${REPAIR_ROBUST_TORSO:-0}"
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0 GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_TOKEN_DROPOUT="${REPAIR_GBT_TOKEN_DROPOUT:-0}"
export GBT_TOKEN_DROPOUT_EPOCHS="${REPAIR_GBT_TOKEN_DROPOUT_EPOCHS:-0}"
export RUMPL_TOKEN_DROPOUT=0
export RUMPL_GBT_SYNTHETIC_REPLACE_PROB="${REPAIR_SYNTHETIC_REPLACE_PROB:-0}"
export RUMPL_GBT_SYNTHETIC_RADIUS_MIN_M="${REPAIR_SYNTHETIC_RADIUS_MIN_M:-3.0}"
export RUMPL_GBT_SYNTHETIC_RADIUS_MAX_M="${REPAIR_SYNTHETIC_RADIUS_MAX_M:-6.0}"
export RUMPL_GBT_SYNTHETIC_HEIGHT_MIN_M="${REPAIR_SYNTHETIC_HEIGHT_MIN_M:-1.0}"
export RUMPL_GBT_SYNTHETIC_HEIGHT_MAX_M="${REPAIR_SYNTHETIC_HEIGHT_MAX_M:-2.5}"
export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0
export RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
export RUMPL_TRAIN_SCOPE=all
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
export RUMPL_VIEW_COUNT_WEIGHTS="${REPAIR_VIEW_COUNT_WEIGHTS:-8,1,1}"

if [[ ! -s "${ROOT}/final_checkpoint.txt" ]]; then
  {
    echo "stage=parallel HRNet canonical repair branch"
    echo "name=${REPAIR_NAME} init=${INIT}"
    echo "physical_gpu=${VISIBLE_GPU} logical_gpu=0"
    echo "epochs=${REPAIR_EPOCHS} lr=${REPAIR_LR} lr_steps=${REPAIR_LR_STEPS}"
    echo "pelvis_prior=${REPAIR_PELVIS_PRIOR} body_reg=${REPAIR_BODY_REG}"
    echo "robust_torso=${REPAIR_ROBUST_TORSO:-0}"
    echo "gbt_token_dropout=${REPAIR_GBT_TOKEN_DROPOUT:-0}"
    echo "gbt_token_dropout_epochs=${REPAIR_GBT_TOKEN_DROPOUT_EPOCHS:-0}"
    echo "view_count_weights=${REPAIR_VIEW_COUNT_WEIGHTS:-8,1,1}"
    echo "synthetic_replace_probability=${REPAIR_SYNTHETIC_REPLACE_PROB:-0}"
    echo "synthetic_radius_m=${REPAIR_SYNTHETIC_RADIUS_MIN_M:-3.0}-${REPAIR_SYNTHETIC_RADIUS_MAX_M:-6.0}"
    echo "synthetic_height_m=${REPAIR_SYNTHETIC_HEIGHT_MIN_M:-1.0}-${REPAIR_SYNTHETIC_HEIGHT_MAX_M:-2.5}"
    echo "selection=predeclared final epoch"
    sha256sum "${CFG}" "${INIT}"
  } >"${ROOT}/manifest.txt"
  cd "${REPO}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
    --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
    --validate-on-two-datasets 0 --use-mmpose-val 1 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${TAG}" \
    >"${ROOT}/train.log" 2>&1
  model_dir=$(find "${MODEL_ROOT}" -maxdepth 1 -type d \
    -name "${TAG}_*" -print | sort | tail -1)
  test -d "${model_dir}"
  checkpoint="${model_dir}/final_state.pth.tar"
  test -s "${checkpoint}"
  printf '%s\n' "${model_dir}" >"${ROOT}/model_dir.txt"
  printf '%s\n' "${checkpoint}" >"${ROOT}/final_checkpoint.txt"
fi

checkpoint=$(cat "${ROOT}/final_checkpoint.txt")
if [[ "${REPAIR_SKIP_FORMAL_EVAL:-0}" == "1" ]]; then
  date --iso-8601=seconds >"${ROOT}/TRAINING_COMPLETED"
  echo "[HRNet repair branch] training complete; formal evaluation deferred ${ROOT}"
  exit 0
fi
for views in 2 3 4; do
  eval_dir="${ROOT}/eval/V${views}"
  if [[ ! -s "${eval_dir}/table2.json" ]]; then
    mkdir -p "${eval_dir}"
    RUMPL_EVAL_STRICT=1 "${PY}" -u "${REPO}/run/eval_rumpl_checkpoint.py" \
      --cfg "${CFG}" --checkpoint "${checkpoint}" \
      --output-dir "${eval_dir}" --workers 8 --gpu 0 \
      --use-mmpose-val true --flip-lower-body-kp-test false \
      --test-on-all-cameras true --n-views-combinations "${views}" \
      --model-num-views 4 --test-mmpose-type "${TYPE}" \
      >"${eval_dir}/eval.log" 2>&1
    pred=$(find "${eval_dir}" -maxdepth 1 \
      -name 'preds_gt_*_dict.pkl' -print -quit)
    test -s "${pred}"
    "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
      --output-json "${eval_dir}/table2.json" \
      >"${eval_dir}/table2.log" 2>&1
  fi
done
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[HRNet repair branch] complete ${ROOT}"
