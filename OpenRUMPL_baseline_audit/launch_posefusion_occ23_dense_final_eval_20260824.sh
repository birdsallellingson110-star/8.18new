#!/usr/bin/env bash
# Evaluate the frozen Stage-1 full baselines and matched Algebraic controls.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
SPARSE_GT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
ROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824
EVAL_ROOT_NAME=${VOC_EVAL_ROOT_NAME:-eval}
FINAL_MARKER=${VOC_FINAL_MARKER:-final_eval_COMPLETED}

HRNET_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
HRNET_SCORER_ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2
HRNET_H18=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h18_clean_temporal_lowlr/model_best.pth.tar
RESNET_BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full
RESNET_SCORER_ROOT=${RESNET_BASE}/e2_c2_identity_hinge
RESNET_H18=${RESNET_BASE}/h18_identity_hinge_v2_gtinput/model_best.pth.tar

for required in "${CFG}" "${GT}" "${HRNET_CKPT}" "${HRNET_H18}" \
  "${HRNET_SCORER_ROOT}/seed0/model_best.pth.tar" \
  "${HRNET_SCORER_ROOT}/seed1/model_best.pth.tar" \
  "${RESNET_BASE}/checkpoint.txt" "${RESNET_H18}" \
  "${RESNET_SCORER_ROOT}/seed0/model_best.pth.tar" \
  "${RESNET_SCORER_ROOT}/seed1/model_best.pth.tar"; do
  test -s "${required}"
done
while [[ ! -s "${ROOT}/frontends_COMPLETED" ]]; do sleep 20; done

set_common_environment() {
  export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
  export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
  export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0
  export GBT_GLOBAL_JV_GATED=0 GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0
  export GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
  export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
  export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
}

run_chain() (
  set -euo pipefail
  local variant=$1 chain=$2 gpu=$3
  local frontend type checkpoint scorer_root scorer h18 out
  set_common_environment
  if [[ "${chain}" == hrnet ]]; then
    frontend=${ROOT}/${variant}/frontends/hrnet/merged/h36m_validation.pkl
    type=posefusion_${variant}_dense_hrnet
    checkpoint=${HRNET_CKPT}
    scorer_root=${HRNET_SCORER_ROOT}
    scorer=${scorer_root}/seed0/model_best.pth.tar
    h18=${HRNET_H18}
    out=${ROOT}/${variant}/${EVAL_ROOT_NAME}/hrnet
    export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
  elif [[ "${chain}" == resnet152 ]]; then
    frontend=${ROOT}/${variant}/frontends/resnet152/h36m_validation.pkl
    type=posefusion_${variant}_dense_resnet152
    checkpoint=$(cat "${RESNET_BASE}/checkpoint.txt")
    scorer_root=${RESNET_SCORER_ROOT}
    scorer=${scorer_root}/seed0/model_best.pth.tar
    h18=${RESNET_H18}
    out=${ROOT}/${variant}/${EVAL_ROOT_NAME}/resnet152
    export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
  else
    echo "unknown chain ${chain}" >&2
    return 2
  fi
  local cache=${out}/cache/validation_22c.npz
  mkdir -p "${out}/cache" "${out}/scores" "${out}/fused" "${out}/results"
  test -s "${frontend}"

  if [[ ! -s "${out}/cache/validation_11c.npz" ]]; then
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
      "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
        --cfg "${CFG}" --checkpoint "${checkpoint}" \
        --dataset-name annot_temporal_5_5 --mmpose-type "${type}" \
        --subset validation --flip-lower-body-kp-test false \
        --output "${out}/cache/validation_11c.npz" \
        --batch-size 256 --workers 6 --gpu 0 >"${out}/cache/export.log" 2>&1
  fi
  if [[ ! -s "${cache}" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${out}/cache/validation_11c.npz" --output "${cache}" \
      >"${out}/cache/append.log" 2>&1
  fi

  "${PY}" -u "${AUDIT}/evaluate_h36m_occl_direct_cache_20260822.py" \
    --cache "${out}/cache/validation_11c.npz" \
    --output "${out}/results/direct_t1.json" >"${out}/results/direct_t1.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
    "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
      --cache "${cache}" --checkpoint-root "${scorer_root}" \
      --output "${out}/results/e2_t1.json" --v2-temperature 0.4 \
      --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
      >"${out}/results/e2_t1.log" 2>&1

  if [[ ! -s "${out}/scores/e2.npy" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
      --cache "${cache}" --checkpoint "${scorer}" --output "${out}/scores/e2.npy" \
      --batch-size 512 --gpu 0 >"${out}/scores/build.log" 2>&1
  fi
  if [[ ! -s "${out}/fused/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
      "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
        --cache "${cache}" --scores "${out}/scores/e2.npy" \
        --output-dir "${out}/fused" --temperature-v2 0.4 \
        --temperature-v3 1.8 --temperature-v4 1.8 --chunk-size 512 --gpu 0 \
        >"${out}/fused/build.log" 2>&1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
    "${AUDIT}/evaluate_frozen_h18_on_occlusion_20260822.py" \
      --validation-cache "${cache}" --validation-fused "${out}/fused/fused_poses.npy" \
      --validation-pkl "${frontend}" --checkpoint "${h18}" \
      --target-centers-pkl "${SPARSE_GT}" \
      --output "${out}/results/final_h18_t9.json" --frame-stride 5 \
      --batch-size 256 --gpu 0 >"${out}/results/final_h18_t9.log" 2>&1

  sha256sum "${frontend}" "${cache}" "${checkpoint}" "${scorer}" "${h18}" \
    >"${out}/sha256.txt"
  date --iso-8601=seconds >"${out}/COMPLETED"
  echo "[${variant} ${chain} final evaluation] complete"
)

# Each dataset owns one GPU.  Its two frozen chains may coexist in memory and
# keep the device busy during the long dense export.
run_chain occ2 resnet152 0 & p0=$!
run_chain occ2 hrnet 0 & p1=$!
run_chain occ3 resnet152 1 & p2=$!
run_chain occ3 hrnet 1 & p3=$!
failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || failed=1; done
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/${FINAL_MARKER}"
echo "[dense VOC Occ-2/Occ-3 final evaluation] complete"
