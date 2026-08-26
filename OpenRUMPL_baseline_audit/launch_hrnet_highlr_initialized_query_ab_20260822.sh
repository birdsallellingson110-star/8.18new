#!/usr/bin/env bash
# Attach global Joint-Query directly to the strongest trainable HRNet V2
# checkpoint (36.885/31.451/30.277), rather than to the balanced but weaker-V2
# C2 model. Both arms preserve VFT/PFT and jointly adapt the complete network.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=gbt_yolox_x_score001_fallback_legswap
INIT=${INIT_OVERRIDE:-/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD2_HIGH_LR5E5_B2_MIXED_T1_20E_seed0_20260815_2026-08-15_18-45-38/model_best.pth.tar}
ROOT=${ROOT_OVERRIDE:-/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/highlr_initialized_query_ab}
SOURCE_RESULT=${SOURCE_RESULT_OVERRIDE:-36.885/31.451/30.277}
TAG_PREFIX=${TAG_PREFIX_OVERRIDE:-HRNET_HIGHINIT_GQ}
A_LR=${A_LR_OVERRIDE:-5e-6}
A_WEIGHTS=${A_WEIGHTS_OVERRIDE:-3,1,1}
B_LR=${B_LR_OVERRIDE:-1e-5}
B_WEIGHTS=${B_WEIGHTS_OVERRIDE:-3,2,2}

mkdir -p "${ROOT}"
test -s "${CFG}"; test -s "${INIT}"
export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

set_common_env() {
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
  export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
  export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
  export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
  export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
  export GBT_TOKEN_DROPOUT=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0
  export RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
  export RUMPL_INIT_CHECKPOINT="${INIT}" RUMPL_TRAIN_SCOPE=all
  export RUMPL_END_EPOCH=12 RUMPL_LR_STEPS=8,10
}

run_one() {
  local name="$1" gpu="$2" lr="$3" weights="$4" tag="$5"
  local out="${ROOT}/${name}"
  mkdir -p "${out}"
  if [[ ! -s "${out}/checkpoint.txt" ]]; then
    (
      set_common_env
      export RUMPL_FINETUNE_LR="${lr}" RUMPL_VIEW_COUNT_WEIGHTS="${weights}"
      {
        echo "init=${INIT} source_result=${SOURCE_RESULT}"
        echo "architecture=H76_VFT_PFT_plus_global_joint_query depth=2 max_delta=0.5"
        echo "epochs=12 lr=${lr} lr_steps=8,10 view_weights=${weights} seed=0"
        sha256sum "${CFG}" "${INIT}"
      } >"${out}/manifest.txt"
      cd "${REPO}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u run/train_rumpl.py \
        --cfg "${CFG}" --gpus 0 --workers 6 --seed 0 \
        --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
        --validate-on-two-datasets 0 --use-mmpose-val 1 \
        --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
        >"${out}/train.log" 2>&1
      ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
      test -s "${ckpt}"; printf '%s\n' "${ckpt}" >"${out}/checkpoint.txt"
    )
  fi
  local ckpt
  ckpt=$(cat "${out}/checkpoint.txt"); test -s "${ckpt}"
  set_common_env
  export RUMPL_FINETUNE_LR="${lr}" RUMPL_VIEW_COUNT_WEIGHTS="${weights}"
  for views in 2 3 4; do
    local eval_dir="${out}/eval/V${views}"
    if [[ ! -s "${eval_dir}/table2.json" ]]; then
      mkdir -p "${eval_dir}"
      cd "${REPO}"
      CUDA_VISIBLE_DEVICES="${gpu}" RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
        --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
        --workers 6 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
        --test-on-all-cameras true --n-views-combinations "${views}" \
        --model-num-views 4 --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1
      pred=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${pred}"
      "${PY}" run/eval_h36m_table2.py --dict-pkl "${pred}" \
        --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
    fi
  done
  date --iso-8601=seconds >"${out}/COMPLETED"
}

run_one A_v2_preserve 0 "${A_LR}" "${A_WEIGHTS}" "${TAG_PREFIX}_A_12E_20260822" & p0=$!
run_one B_multiview_recover 1 "${B_LR}" "${B_WEIGHTS}" "${TAG_PREFIX}_B_12E_20260822" & p1=$!
wait "${p0}" "${p1}"
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[HRNet high-LR initialized Joint-Query A/B] complete"
