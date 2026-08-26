#!/usr/bin/env bash
# ResNet V2 repair without sacrificing V3/V4.
# A: long-budget mixed-cardinality H76 + geometry uncertainty.
# B: full-network GBT/MVGFormer-style joint-query residual screen, queued after
#    the active geometry-E2 H18 run releases GPU1.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=res152_lt_alg_undistorted_annbox
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair
H18_DONE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/h18_geom_e2/COMPLETED
mkdir -p "${OUT}"
test -s "${CFG}"; test -s "${FRONTEND}/train/h36m_train_res152.pkl"
test -s "${FRONTEND}/validation/h36m_validation_res152.pkl"

set_common_env() {
  export PYTHONPATH=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
  export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FINETUNE_LR=1e-4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
  export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0
  export GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
  export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0
  export BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all
}

evaluate_checkpoint() {
  local checkpoint="$1" root="$2" gpu="$3"
  test -s "${checkpoint}"
  for views in 2 3 4; do
    local eval_dir="${root}/eval/V${views}"
    mkdir -p "${eval_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" RUMPL_EVAL_STRICT=1 "${PY}" -u \
      "${REPO}/run/eval_rumpl_checkpoint.py" \
      --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
      --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
      --test-on-all-cameras true --n-views-combinations "${views}" \
      --model-num-views 4 --test-mmpose-type "${TYPE}" \
      >"${eval_dir}/eval.log" 2>&1
    local prediction
    prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    test -n "${prediction}"; test -s "${prediction}"
    "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${prediction}" \
      --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
  done
}

run_long_mixed() {
  local root="${OUT}/A_long_mixed_geom" tag
  tag=RES152_V2REPAIR_LONGMIXED_GEOM_123E_3to1to1_seed0_20260822
  mkdir -p "${root}"
  [[ -s "${root}/COMPLETED" ]] && return 0
  (
    set_common_env
    export CUDA_VISIBLE_DEVICES=0
    export RUMPL_END_EPOCH=123 RUMPL_LR_STEPS=62,93
    export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=1
    export RUMPL_GBT_QUERY_RESIDUAL=0
    {
      echo "method=long mixed-cardinality ResNet H76 geometry uncertainty"
      echo "epochs=123 approx_updates=299874 ratio=3,1,1 fixed_K2_epochs=8"
      echo "goal=V2/V3/V4 simultaneous decrease; input=${TYPE}"
      sha256sum "${CFG}" "${FRONTEND}/train/h36m_train_res152.pkl" \
        "${FRONTEND}/validation/h36m_validation_res152.pkl"
    } >"${root}/train.log"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
      --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >>"${root}/train.log" 2>&1
    local checkpoint
    checkpoint=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -n "${checkpoint}"; test -s "${checkpoint}"
    printf '%s\n' "${checkpoint}" >"${root}/checkpoint.txt"
    evaluate_checkpoint "${checkpoint}" "${root}" 0
    date --iso-8601=seconds >"${root}/COMPLETED"
  )
}

run_query_screen() {
  local root="${OUT}/B_global_query_full" tag
  tag=RES152_V2REPAIR_GBTQUERY_FULL_20E_3to1to1_seed0_20260822
  mkdir -p "${root}"
  [[ -s "${root}/COMPLETED" ]] && return 0
  while [[ ! -s "${H18_DONE}" ]]; do sleep 30; done
  (
    set_common_env
    export CUDA_VISIBLE_DEVICES=1
    export RUMPL_END_EPOCH=20
    unset RUMPL_LR_STEPS
    export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
    export RUMPL_GBT_QUERY_RESIDUAL=1
    export RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
    export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
    export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
    {
      echo "method=full-network global joint-query residual on ResNet rays"
      echo "epochs=20 ratio=3,1,1 fixed_K2_epochs=8 frozen=none"
      echo "difference_from_failed_E3=fresh_full_network_training_and_ResNet_input"
      sha256sum "${CFG}" "${FRONTEND}/train/h36m_train_res152.pkl" \
        "${FRONTEND}/validation/h36m_validation_res152.pkl"
    } >"${root}/train.log"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
      --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >>"${root}/train.log" 2>&1
    local checkpoint
    checkpoint=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -n "${checkpoint}"; test -s "${checkpoint}"
    printf '%s\n' "${checkpoint}" >"${root}/checkpoint.txt"
    evaluate_checkpoint "${checkpoint}" "${root}" 1
    date --iso-8601=seconds >"${root}/COMPLETED"
  )
}

run_long_mixed & p0=$!
run_query_screen & p1=$!
wait "${p0}" "${p1}"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet V2 repair pair] complete"
