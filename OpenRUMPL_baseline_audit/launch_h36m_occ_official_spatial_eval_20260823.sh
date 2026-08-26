#!/usr/bin/env bash
# Evaluate the frozen clean-trained HRNet/ResNet Stage-I chains on Occ-2/Occ-3.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/h36m_occ_official_20260823
OCC2_VARIANT=${OCC2_VARIANT:-occ2}
OCC3_VARIANT=${OCC3_VARIANT:-occ3}

HRNET_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
HRNET_SCORER=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_training_protocol_v2
RESNET_BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full
RESNET_SCORER=${RESNET_BASE}/e2_c2_identity_hinge

for required in "${CFG}" "${HRNET_CKPT}" \
  "${HRNET_SCORER}/seed0/model_best.pth.tar" \
  "${HRNET_SCORER}/seed1/model_best.pth.tar" \
  "${RESNET_BASE}/checkpoint.txt" \
  "${RESNET_SCORER}/seed0/model_best.pth.tar" \
  "${RESNET_SCORER}/seed1/model_best.pth.tar"; do
  test -s "${required}"
done

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
  local variant=$1
  local chain=$2
  local gpu=$3
  local frontend type checkpoint scorer out
  local v2_temp v34_temp

  set_common_environment
  if [[ "${chain}" == hrnet ]]; then
    frontend=${ROOT}/${variant}/frontends/hrnet/merged/h36m_validation.pkl
    type=h36m_${variant}_official_hrnet_gbt_aligned
    checkpoint=${HRNET_CKPT}
    scorer=${HRNET_SCORER}
    out=${ROOT}/${variant}/eval/hrnet_spatial
    v2_temp=0.4
    v34_temp=1.8
    export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
  elif [[ "${chain}" == resnet152 ]]; then
    frontend=${ROOT}/${variant}/frontends/resnet152/h36m_validation.pkl
    type=h36m_${variant}_official_res152_lt
    checkpoint=$(cat "${RESNET_BASE}/checkpoint.txt")
    scorer=${RESNET_SCORER}
    out=${ROOT}/${variant}/eval/resnet152_spatial
    v2_temp=0.4
    v34_temp=1.8
    export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
  else
    echo "unknown chain ${chain}" >&2
    return 2
  fi

  local dataset=annot_filtered_5_64
  local type_dir=${DATA}/data/datasets_mmpose/${dataset}_${type}
  test -s "${frontend}"
  test -L "${type_dir}/h36m_validation.pkl" -o -s "${type_dir}/h36m_validation.pkl"
  mkdir -p "${out}"

  if [[ ! -s "${out}/validation_11c.npz" ]]; then
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
      "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
        --cfg "${CFG}" --checkpoint "${checkpoint}" \
        --dataset-name "${dataset}" --mmpose-type "${type}" --subset validation \
        --flip-lower-body-kp-test false --output "${out}/validation_11c.npz" \
        --batch-size 256 --workers 6 --gpu 0 >"${out}/export_11c.log" 2>&1
  fi

  "${PY}" -u "${AUDIT}/evaluate_h36m_occl_direct_cache_20260822.py" \
    --cache "${out}/validation_11c.npz" --output "${out}/direct_result.json" \
    >"${out}/direct_result.log" 2>&1

  if [[ ! -s "${out}/validation_22c.npz" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${out}/validation_11c.npz" --output "${out}/validation_22c.npz" \
      >"${out}/append_22c.log" 2>&1
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
    "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
      --cache "${out}/validation_22c.npz" --checkpoint-root "${scorer}" \
      --output "${out}/e2_result.json" --v2-temperature "${v2_temp}" \
      --v3-temperature "${v34_temp}" --v4-temperature "${v34_temp}" \
      --batch-size 1024 --gpu 0 >"${out}/e2_result.log" 2>&1

  sha256sum "${frontend}" "${out}/validation_11c.npz" \
    "${out}/validation_22c.npz" "${checkpoint}" \
    "${scorer}/seed0/model_best.pth.tar" \
    "${scorer}/seed1/model_best.pth.tar" >"${out}/sha256.txt"
  date --iso-8601=seconds >"${out}/COMPLETED"
  echo "[${variant} ${chain} spatial] complete"
)

test -s "${ROOT}/frontends_COMPLETED"
run_chain "${OCC2_VARIANT}" hrnet 0 & p0=$!
run_chain "${OCC2_VARIANT}" resnet152 0 & p1=$!
run_chain "${OCC3_VARIANT}" hrnet 1 & p2=$!
run_chain "${OCC3_VARIANT}" resnet152 1 & p3=$!
failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || failed=1; done
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/spatial_eval_COMPLETED"
echo "[Human3.6M-Occluded official spatial evaluation] complete"
