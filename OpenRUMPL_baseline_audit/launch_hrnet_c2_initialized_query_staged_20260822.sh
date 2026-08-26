#!/usr/bin/env bash
# Preserve the established HRNet C2 spatial solution while attaching the same
# global Joint-Query residual used by the ResNet line.  Stage U1 trains only
# the zero-initialized query adapter; stage U2 optionally fine-tunes the whole
# network at a much lower LR.  Both stages are evaluated independently so a
# degrading joint fine-tune can never replace the preserved U1 result.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
C2_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
GPU=${1:-1}
ROOT=/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/c2_initialized_query_staged

mkdir -p "${ROOT}"
test -s "${CFG}"; test -s "${C2_CKPT}"; test -s "${TRAIN_FRONTEND}"; test -s "${VAL_FRONTEND}"

export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

set_common_env() {
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
  export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
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
}

evaluate_stage() {
  local stage_root="$1" ckpt="$2"
  for views in 2 3 4; do
    local eval_dir="${stage_root}/eval/V${views}"
    if [[ ! -s "${eval_dir}/table2.json" ]]; then
      mkdir -p "${eval_dir}"
      cd "${REPO}"
      CUDA_VISIBLE_DEVICES="${GPU}" RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
        --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
        --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
        --test-on-all-cameras true --n-views-combinations "${views}" \
        --model-num-views 4 --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1
      local pred
      pred=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${pred}"
      "${PY}" run/eval_h36m_table2.py --dict-pkl "${pred}" \
        --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
    fi
  done
}

run_stage() {
  local name="$1" init="$2" scope="$3" epochs="$4" lr="$5" tag="$6"
  local stage_root="${ROOT}/${name}"
  mkdir -p "${stage_root}"
  set_common_env
  export RUMPL_INIT_CHECKPOINT="${init}" RUMPL_TRAIN_SCOPE="${scope}"
  export RUMPL_END_EPOCH="${epochs}" RUMPL_FINETUNE_LR="${lr}"
  if [[ ! -s "${stage_root}/checkpoint.txt" ]]; then
    {
      echo "stage=${name} init=${init} scope=${scope} epochs=${epochs} lr=${lr}"
      echo "view_count_weights=8,1,1 frontend=HRNet-W32 coordinates/confidence"
      sha256sum "${CFG}" "${init}" "${TRAIN_FRONTEND}" "${VAL_FRONTEND}"
    } >"${stage_root}/manifest.txt"
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u run/train_rumpl.py \
      --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
      --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >"${stage_root}/train.log" 2>&1
    local ckpt
    ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -s "${ckpt}"; printf '%s\n' "${ckpt}" >"${stage_root}/checkpoint.txt"
  fi
  local stage_ckpt
  stage_ckpt=$(cat "${stage_root}/checkpoint.txt"); test -s "${stage_ckpt}"
  evaluate_stage "${stage_root}" "${stage_ckpt}"
  date --iso-8601=seconds >"${stage_root}/COMPLETED"
}

# U1 cannot damage C2 because only the new query residual parameters train.
run_stage U1_query_only "${C2_CKPT}" gbt_query_residual 8 1e-4 \
  HRNET_C2INIT_GQ_U1_QUERYONLY_8E_seed0_20260822

# U2 tests whether a conservative end-to-end adjustment helps.  U1 remains a
# separately reported checkpoint and is retained if this stage regresses.
U1_CKPT=$(cat "${ROOT}/U1_query_only/checkpoint.txt")
run_stage U2_all_low_lr "${U1_CKPT}" all 12 5e-6 \
  HRNET_C2INIT_GQ_U2_ALL_12E_LR5e6_seed0_20260822

date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[HRNet C2-initialized staged Joint-Query] complete"
