#!/usr/bin/env bash
# Repair phase 1: continue the canonical HRNet generator from the fixed
# 20-epoch endpoint at a low learning rate.  The endpoint, rather than the
# internal four-view model_best checkpoint, is used for the formal evaluation.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
INIT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CAMGEN_STAGE1_hrnet_20E_seed0_20260824_2026-08-24_19-04-45/final_state.pth.tar
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/phase1_cont20_lr1e5
TAG=CAMGEN_HRNET_CANON_REPAIR_CONT20E_LR1E5_seed0_20260824

TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl

mkdir -p "${ROOT}"
test -s "${CFG}"
test -s "${INIT}"
test -s "${TRAIN_PKL}"
test -s "${VAL_PKL}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_END_EPOCH=20
export RUMPL_INIT_CHECKPOINT="${INIT}"
export RUMPL_FINETUNE_LR=1e-5 RUMPL_LR_STEPS=10,15
export RUMPL_SAVE_EVERY_N_EPOCHS=1
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_BODY_CANONICAL_FRAME=1 RUMPL_BODY_CANONICAL_REG=1e-4
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
export GBT_TOKEN_DROPOUT=0 RUMPL_TOKEN_DROPOUT=0
export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0
export RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
export RUMPL_TRAIN_SCOPE=all
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1

if [[ ! -s "${ROOT}/final_checkpoint.txt" ]]; then
  {
    echo "stage=HRNet canonical generator repair phase 1"
    echo "policy=predeclared final epoch; never select with V2/V3/V4 test MPJPE"
    echo "init=${INIT}"
    echo "epochs=20 lr=1e-5 lr_steps=10,15 view_weights=8,1,1 seed=0"
    echo "body_canonical=1 query=0 save_every_epoch=1"
    echo "train=${TRAIN_PKL}"
    echo "validation=${VAL_PKL}"
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
  final_checkpoint="${model_dir}/final_state.pth.tar"
  test -s "${final_checkpoint}"
  printf '%s\n' "${model_dir}" >"${ROOT}/model_dir.txt"
  printf '%s\n' "${final_checkpoint}" >"${ROOT}/final_checkpoint.txt"
fi

checkpoint=$(cat "${ROOT}/final_checkpoint.txt")
test -s "${checkpoint}"
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
echo "[HRNet canonical repair phase 1] complete ${ROOT}"
