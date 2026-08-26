#!/usr/bin/env bash
# Stage-1 clean H36M generator retraining after removing world-frame
# dependence.  HRNet runs first on GPU 0; the matched ResNet line is queued
# behind it because GPU 1 currently belongs to another user's experiment.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999

HRNET_TYPE=gbt_yolox_x_score001_fallback_legswap
HRNET_TRAIN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
HRNET_VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
RESNET_TYPE=res152_lt_alg_undistorted_annbox
RESNET_TRAIN=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/train/h36m_train_res152.pkl
RESNET_VAL=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/validation/h36m_validation_res152.pkl

mkdir -p "${ROOT}"
test -s "${CFG}"
test -s "${HRNET_TRAIN}"; test -s "${HRNET_VAL}"
test -s "${RESNET_TRAIN}"; test -s "${RESNET_VAL}"

set_common_env() {
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="${AUDIT}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_END_EPOCH=20
  export RUMPL_FINETUNE_LR=1e-4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
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
  export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0
  export RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
  export RUMPL_TRAIN_SCOPE=all
  unset RUMPL_INIT_CHECKPOINT
}

run_generator() {
  local name=$1 type=$2 train_pkl=$3 val_pkl=$4 query=$5 schedule=$6
  local dir="${ROOT}/${name}/generator"
  local tag="CAMGEN_STAGE1_${name}_20E_seed0_20260824"
  mkdir -p "${dir}"
  set_common_env

  if [[ "${schedule}" == k2_then_mixed ]]; then
    export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
    export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
  else
    unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
    export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1
  fi
  if [[ "${query}" == 1 ]]; then
    export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
    export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
    export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
  else
    export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
    export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
    export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
  fi

  if [[ ! -s "${dir}/checkpoint.txt" ]]; then
    {
      echo "stage=H36M clean camera-independent generator"
      echo "frontend=${name} type=${type} query=${query} schedule=${schedule}"
      echo "subjects_train=S1,S5,S6,S7,S8 subjects_test=S9,S11"
      echo "epochs=20 lr=1e-4 body_canonical=1 seed=0"
      echo "train=${train_pkl}"
      echo "validation=${val_pkl}"
      sha256sum "${CFG}"
    } >"${dir}/manifest.txt"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py \
      --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
      --train-mmpose-type "${type}" --test-mmpose-type "${type}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >"${dir}/train.log" 2>&1
    ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f \
      -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -s "${ckpt}"
    printf '%s\n' "${ckpt}" >"${dir}/checkpoint.txt"
  fi

  local ckpt
  ckpt=$(cat "${dir}/checkpoint.txt")
  test -s "${ckpt}"
  for views in 2 3 4; do
    local eval_dir="${dir}/eval/V${views}"
    if [[ ! -s "${eval_dir}/table2.json" ]]; then
      mkdir -p "${eval_dir}"
      RUMPL_EVAL_STRICT=1 "${PY}" -u "${REPO}/run/eval_rumpl_checkpoint.py" \
        --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
        --workers 8 --gpu 0 --use-mmpose-val true \
        --flip-lower-body-kp-test false --test-on-all-cameras true \
        --n-views-combinations "${views}" --model-num-views 4 \
        --test-mmpose-type "${type}" >"${eval_dir}/eval.log" 2>&1
      pred=$(find "${eval_dir}" -maxdepth 1 \
        -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${pred}"
      "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
        --output-json "${eval_dir}/table2.json" \
        >"${eval_dir}/table2.log" 2>&1
    fi
  done
  date --iso-8601=seconds >"${dir}/COMPLETED"
}

run_generator hrnet "${HRNET_TYPE}" "${HRNET_TRAIN}" "${HRNET_VAL}" 0 k2_heavy
run_generator resnet152 "${RESNET_TYPE}" "${RESNET_TRAIN}" "${RESNET_VAL}" 1 k2_then_mixed
date --iso-8601=seconds >"${ROOT}/GENERATORS_COMPLETED"
echo "[stage1 canonical generators] complete ${ROOT}"
