#!/usr/bin/env bash
# Continue Stage-1 after both camera-independent generators finish:
# 22-candidate export -> canonical E2 identity fusion -> camera-independent H18.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend

set_common_env() {
  export CUDA_VISIBLE_DEVICES="${STAGE1_VISIBLE_GPU:-0}"
  export PYTHONPATH="${AUDIT}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_BODY_CANONICAL_FRAME=1
  export RUMPL_BODY_CANONICAL_REG="${RUMPL_BODY_CANONICAL_REG:-1e-4}"
  export RUMPL_BODY_CANONICAL_PELVIS_PRIOR="${RUMPL_BODY_CANONICAL_PELVIS_PRIOR:-0}"
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
  export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_LEARNABLE_BIAS=0
}

configure_query() {
  local enabled=$1
  export RUMPL_GBT_QUERY_RESIDUAL="${enabled}"
  export RUMPL_GBT_QUERY_RESIDUAL_GLOBAL="${enabled}"
  export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
  export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
}

run_frontend() {
  local name=$1 type=$2 query=$3 train_pkl=$4 val_pkl=$5
  local temporal_dataset=$6 temporal_pkl=$7
  local base="${ROOT}/${name}"
  local generator="${base}/generator"
  local cache="${base}/canonical_e2/cache"
  local scorer="${base}/canonical_e2/identity_hinge"
  local temporal_cache="${base}/canonical_h18/cache"
  local scores="${base}/canonical_h18/scores"
  local fused="${base}/canonical_h18/fused"
  local temporal_out="${base}/canonical_h18/model"
  mkdir -p "${cache}" "${scorer}" "${temporal_cache}" \
    "${scores}" "${fused}" "${temporal_out}"
  local ckpt
  ckpt=$(cat "${generator}/checkpoint.txt")
  test -s "${ckpt}"
  set_common_env
  configure_query "${query}"

  export_cache() {
    local split=$1 dataset_name=$2 output=$3
    [[ -s "${output}" ]] && return 0
    cd "${REPO}"
    "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
      --cfg "${CFG}" --checkpoint "${ckpt}" --dataset-name "${dataset_name}" \
      --mmpose-type "${type}" --subset "${split}" \
      --flip-lower-body-kp-test false --output "${output}" \
      --batch-size 256 --workers 8 --gpu "${STAGE1_VISIBLE_GPU:-0}" \
      >"${output%.npz}.log" 2>&1
  }

  export_cache train annot_filtered_5_64 "${cache}/train_11c.npz"
  export_cache validation annot_filtered_5_64 "${cache}/validation_11c.npz"
  for split in train validation; do
    if [[ ! -s "${cache}/${split}_22c.npz" ]]; then
      local append_args=()
      if [[ -n "${CANDIDATE_SOLVER_MODE:-}" ]]; then
        append_args+=(--solver-mode "${CANDIDATE_SOLVER_MODE}")
      fi
      if [[ -n "${CANDIDATE_BLEND_ALPHA:-}" ]]; then
        append_args+=(--blend-alpha "${CANDIDATE_BLEND_ALPHA}")
      fi
      if [[ -n "${CANDIDATE_MAX_DELTA_M:-}" ]]; then
        append_args+=(--max-delta-m "${CANDIDATE_MAX_DELTA_M}")
      fi
      "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
        --input "${cache}/${split}_11c.npz" \
        --output "${cache}/${split}_22c.npz" "${append_args[@]}" \
        >"${cache}/append_${split}.log" 2>&1
    fi
  done
  local train_cache="${cache}/train_22c.npz"
  local val_cache="${cache}/validation_22c.npz"

  train_scorer() {
    local seed=$1 dir="${scorer}/seed${1}"
    [[ -s "${dir}/result.json" ]] && return 0
    mkdir -p "${dir}"
    "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
      --train-shards "${train_cache}" --validation-cache "${val_cache}" \
      --output-dir "${dir}" --pretrain-epochs 10 --finetune-epochs 5 \
      --batch-size 128 --temperature 1.8 --target-temperature-mm 5.0 \
      --oracle-weight 1.0 --identity-hinge 0.25 --identity-v2-weight 4.0 \
      --canonical-geometry --fixed-metric-normalization --stage-heads \
      --workers 0 --seed "${seed}" --gpu 0 >"${dir}/train.log" 2>&1
  }
  train_scorer 0 & s0=$!
  train_scorer 1 & s1=$!
  wait "${s0}" "${s1}"
  if [[ ! -s "${scorer}/calibrated_v2t04.json" ]]; then
    "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
      --cache "${val_cache}" --checkpoint-root "${scorer}" \
      --output "${scorer}/calibrated_v2t04.json" --v2-temperature 0.4 \
      --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
      >"${scorer}/calibration.log" 2>&1
  fi

  export_cache validation "${temporal_dataset}" \
    "${temporal_cache}/validation_11c.npz"
  if [[ ! -s "${temporal_cache}/validation_22c.npz" ]]; then
    local append_args=()
    if [[ -n "${CANDIDATE_SOLVER_MODE:-}" ]]; then
      append_args+=(--solver-mode "${CANDIDATE_SOLVER_MODE}")
    fi
    if [[ -n "${CANDIDATE_BLEND_ALPHA:-}" ]]; then
      append_args+=(--blend-alpha "${CANDIDATE_BLEND_ALPHA}")
    fi
    if [[ -n "${CANDIDATE_MAX_DELTA_M:-}" ]]; then
      append_args+=(--max-delta-m "${CANDIDATE_MAX_DELTA_M}")
    fi
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${temporal_cache}/validation_11c.npz" \
      --output "${temporal_cache}/validation_22c.npz" "${append_args[@]}" \
      >"${temporal_cache}/append_validation.log" 2>&1
  fi
  local temporal_val_cache="${temporal_cache}/validation_22c.npz"

  for split in train validation; do
    local source_cache="${train_cache}"
    local score_file="${scores}/train_e2_scores.npy"
    [[ "${split}" == validation ]] && source_cache="${temporal_val_cache}" \
      && score_file="${scores}/validation_e2_scores.npy"
    if [[ ! -s "${score_file}" ]]; then
      "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
        --cache "${source_cache}" \
        --checkpoint "${scorer}/seed0/model_best.pth.tar" \
        --output "${score_file}" --batch-size 256 --gpu 0 \
        >"${scores}/${split}.log" 2>&1
    fi
    if [[ ! -s "${fused}/${split}/manifest.json" ]]; then
      mkdir -p "${fused}/${split}"
      "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
        --cache "${source_cache}" --scores "${score_file}" \
        --output-dir "${fused}/${split}" --temperature-v2 0.4 \
        --temperature-v3 1.8 --temperature-v4 1.8 \
        --chunk-size 256 --gpu 0 >"${fused}/${split}.log" 2>&1
    fi
  done

  if [[ ! -s "${temporal_out}/result.json" ]]; then
    "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
      --train-cache "${train_cache}" \
      --train-fused "${fused}/train/fused_poses.npy" \
      --train-pkl "${train_pkl}" --validation-cache "${temporal_val_cache}" \
      --validation-fused "${fused}/validation/fused_poses.npy" \
      --validation-pkl "${temporal_pkl}" --output-dir "${temporal_out}" \
      --window-length 9 --frame-stride 5 --epochs 12 --batch-size 64 \
      --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 \
      --residual-scale-m 0.10 --camera-independent --gpu 0 --seed 0 \
      >"${temporal_out}/train.log" 2>&1
  fi
  date --iso-8601=seconds >"${base}/STAGE1_COMPLETED"
}

main() {
  while [[ ! -s "${ROOT}/GENERATORS_COMPLETED" ]]; do sleep 30; done

  run_frontend hrnet gbt_yolox_x_score001_fallback_legswap 0 \
    /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
    /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl \
    annot_temporal_5_5 \
    /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl

  run_frontend resnet152 res152_lt_alg_undistorted_annbox 1 \
    /mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/train/h36m_train_res152.pkl \
    /mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/validation/h36m_validation_res152.pkl \
    annot_filtered_5_64_res152_query_temporal_v2_gtinput \
    /mnt/data/cjyoutput/gbt_aligned_resnet_20260822/frontend_temporal_v2_gtinput/validation/h36m_validation_res152_temporal.pkl

  date --iso-8601=seconds >"${ROOT}/STAGE1_COMPLETED"
  echo "[stage1 canonical downstream] complete ${ROOT}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
