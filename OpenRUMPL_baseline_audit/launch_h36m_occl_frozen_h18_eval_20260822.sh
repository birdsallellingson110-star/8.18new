#!/usr/bin/env bash
# Build dense frozen spatial caches and evaluate clean-trained centered H18.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015

while [[ ! -s "${ROOT}/dense_frontends_COMPLETED" ]]; do sleep 20; done

run_hrnet() (
  set -euo pipefail
  local frontend=${ROOT}/hrnet_temporal/frontend/merged/h36m_validation_hrnet_occl_temporal.pkl
  local type=gbt_h36m_occl_f015_seed20260822_hrnet_temporal
  local out=${ROOT}/hrnet_temporal
  local cache=${out}/cache/validation_22c.npz
  local ckpt=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
  local scorer=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2/seed0/model_best.pth.tar
  local h18=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/model_best.pth.tar
  mkdir -p "${out}/cache" "${out}/scores" "${out}/fused" "${out}/evaluation"
  test -s "${frontend}" && test -s "${ckpt}" && test -s "${scorer}" && test -s "${h18}"
  export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
  export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
  export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0 RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
  export GBT_TOKEN_DROPOUT=0
  if [[ ! -s "${out}/cache/validation_11c.npz" ]]; then
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
      --cfg "${CFG}" --checkpoint "${ckpt}" --dataset-name annot_temporal_5_5 \
      --mmpose-type "${type}" --subset validation --flip-lower-body-kp-test false \
      --output "${out}/cache/validation_11c.npz" --batch-size 256 --workers 8 --gpu 0 \
      >"${out}/cache/export.log" 2>&1
  fi
  if [[ ! -s "${cache}" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${out}/cache/validation_11c.npz" --output "${cache}" \
      >"${out}/cache/append.log" 2>&1
  fi
  if [[ ! -s "${out}/scores/e2.npy" ]]; then
    CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
      --cache "${cache}" --checkpoint "${scorer}" --output "${out}/scores/e2.npy" \
      --batch-size 256 --gpu 0 >"${out}/scores/build.log" 2>&1
  fi
  if [[ ! -s "${out}/fused/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
      --cache "${cache}" --scores "${out}/scores/e2.npy" --output-dir "${out}/fused" \
      --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
      --chunk-size 256 --gpu 0 >"${out}/fused/build.log" 2>&1
  fi
  CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/evaluate_frozen_h18_on_occlusion_20260822.py" \
    --validation-cache "${cache}" --validation-fused "${out}/fused/fused_poses.npy" \
    --validation-pkl "${frontend}" --checkpoint "${h18}" \
    --output "${out}/evaluation/centered_h18_result.json" --frame-stride 5 \
    --batch-size 128 --gpu 0 >"${out}/evaluation/centered_h18.log" 2>&1
  date --iso-8601=seconds >"${out}/evaluation/COMPLETED"
)

run_resnet() (
  set -euo pipefail
  local frontend=${ROOT}/resnet_temporal/frontend/h36m_validation_res152_occl_temporal.pkl
  local type=res152_lt_alg_undistorted_annbox_occl_f015_seed20260822_temporal
  local out=${ROOT}/resnet_temporal
  local cache=${out}/cache/validation_22c.npz
  local base=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full
  local ckpt
  ckpt=$(cat "${base}/checkpoint.txt")
  local scorer=${base}/e2_c2_identity_hinge/seed0/model_best.pth.tar
  local h18=${base}/h18_identity_hinge_v2_gtinput/model_best.pth.tar
  mkdir -p "${out}/cache" "${out}/scores" "${out}/fused" "${out}/evaluation"
  test -s "${frontend}" && test -s "${ckpt}" && test -s "${scorer}" && test -s "${h18}"
  export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
  export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
  export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
  export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 RUMPL_TOKEN_DROPOUT=0
  if [[ ! -s "${out}/cache/validation_11c.npz" ]]; then
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
      --cfg "${CFG}" --checkpoint "${ckpt}" --dataset-name annot_temporal_5_5 \
      --mmpose-type "${type}" --subset validation --flip-lower-body-kp-test false \
      --output "${out}/cache/validation_11c.npz" --batch-size 256 --workers 8 --gpu 0 \
      >"${out}/cache/export.log" 2>&1
  fi
  if [[ ! -s "${cache}" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${out}/cache/validation_11c.npz" --output "${cache}" \
      >"${out}/cache/append.log" 2>&1
  fi
  if [[ ! -s "${out}/scores/e2.npy" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
      --cache "${cache}" --checkpoint "${scorer}" --output "${out}/scores/e2.npy" \
      --batch-size 256 --gpu 0 >"${out}/scores/build.log" 2>&1
  fi
  if [[ ! -s "${out}/fused/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
      --cache "${cache}" --scores "${out}/scores/e2.npy" --output-dir "${out}/fused" \
      --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
      --chunk-size 256 --gpu 0 >"${out}/fused/build.log" 2>&1
  fi
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/evaluate_frozen_h18_on_occlusion_20260822.py" \
    --validation-cache "${cache}" --validation-fused "${out}/fused/fused_poses.npy" \
    --validation-pkl "${frontend}" --checkpoint "${h18}" \
    --output "${out}/evaluation/centered_h18_result.json" --frame-stride 5 \
    --batch-size 128 --gpu 0 >"${out}/evaluation/centered_h18.log" 2>&1
  date --iso-8601=seconds >"${out}/evaluation/COMPLETED"
)

run_hrnet & hrnet_pid=$!
run_resnet & resnet_pid=$!
failed=0
wait "${hrnet_pid}" || failed=1
wait "${resnet_pid}" || failed=1
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/frozen_h18_COMPLETED"
echo "[H36M-Occl frozen centered H18] complete"
