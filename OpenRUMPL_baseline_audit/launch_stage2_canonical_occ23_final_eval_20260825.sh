#!/usr/bin/env bash
# Frozen canonical/token10 Stage-1 chains on dense VOC Occ-2/Occ-3.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
SPARSE_GT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
ROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824
CAMROOT=/mnt/data/cjyoutput/camera_generalization_20260824
SELECTION=${CAMROOT}/final_temporal_selection_20260825.json
OCC2_GPU=${STAGE2_OCC2_GPU:-0}
OCC3_GPU=${STAGE2_OCC3_GPU:-1}

HRBASE=${CAMROOT}/hrnet_token10_generalization_20260825
HRNET_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CAMGEN_HRNET_CANON_REPAIR_camera_ab4_token10_synth0_seed0_20260825_2026-08-25_16-44-00/final_state.pth.tar
HRNET_SCORER_ROOT=${HRBASE}/canonical_e2/identity_hinge
HRNET_SCORER=${HRNET_SCORER_ROOT}/seed1/model_best.pth.tar
RNBASE=${CAMROOT}/stage1_h36m_dual_frontend/resnet152
RESNET_CKPT=$(cat "${RNBASE}/generator/checkpoint.txt")
RESNET_SCORER_ROOT=${RNBASE}/canonical_e2/identity_hinge
RESNET_SCORER=${RESNET_SCORER_ROOT}/seed0/model_best.pth.tar

selected_checkpoint() {
  "${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["frontends"][sys.argv[2]]["selected_checkpoint"])' \
    "${SELECTION}" "$1"
}

while [[ ! -s "${ROOT}/frontends_COMPLETED" || ! -s "${SELECTION}" ]]; do sleep 30; done
HRNET_H18=$(selected_checkpoint hrnet)
RESNET_H18=$(selected_checkpoint resnet152)

set_common_environment() {
  export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_BODY_CANONICAL_FRAME=1
  export RUMPL_BODY_CANONICAL_ROBUST_TORSO=0
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
  export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_LEARNABLE_BIAS=0
  export GBT_TOKEN_DROPOUT=0
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
    scorer=${HRNET_SCORER}
    h18=${HRNET_H18}
    out=${ROOT}/${variant}/eval/hrnet
    export RUMPL_BODY_CANONICAL_REG=1e-2 RUMPL_BODY_CANONICAL_PELVIS_PRIOR=1
    export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
  else
    frontend=${ROOT}/${variant}/frontends/resnet152/h36m_validation.pkl
    type=posefusion_${variant}_dense_resnet152
    checkpoint=${RESNET_CKPT}
    scorer_root=${RESNET_SCORER_ROOT}
    scorer=${RESNET_SCORER}
    h18=${RESNET_H18}
    out=${ROOT}/${variant}/eval/resnet152
    export RUMPL_BODY_CANONICAL_REG=1e-4 RUMPL_BODY_CANONICAL_PELVIS_PRIOR=0
    export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
    export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
  fi
  local cache=${out}/cache/validation_22c.npz
  mkdir -p "${out}/cache" "${out}/scores" "${out}/fused" "${out}/uncertainty" "${out}/results"
  test -s "${frontend}" -a -s "${checkpoint}" -a -s "${scorer}" -a -s "${h18}"
  if [[ -s "${out}/COMPLETED" && -s "${out}/results/final_h18_t9.json" ]]; then
    echo "[stage2 ${variant} ${chain}] already complete"
    return 0
  fi

  if [[ ! -s "${out}/cache/validation_11c.npz" ]]; then
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
      "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
      --cfg "${CFG}" --checkpoint "${checkpoint}" --dataset-name annot_temporal_5_5 \
      --mmpose-type "${type}" --subset validation --flip-lower-body-kp-test false \
      --output "${out}/cache/validation_11c.npz" --batch-size 128 \
      --workers 4 --gpu 0 >"${out}/cache/export.log" 2>&1
  fi
  if [[ ! -s "${cache}" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${out}/cache/validation_11c.npz" --output "${cache}" \
      >"${out}/cache/append.log" 2>&1
  fi
  "${PY}" -u "${AUDIT}/evaluate_h36m_occl_direct_cache_20260822.py" \
    --cache "${out}/cache/validation_11c.npz" \
    --output "${out}/results/direct_t1.json" >"${out}/results/direct_t1.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${cache}" --checkpoint-root "${scorer_root}" \
    --output "${out}/results/e2_t1.json" --v2-temperature 0.4 \
    --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 512 --gpu 0 \
    >"${out}/results/e2_t1.log" 2>&1
  if [[ ! -s "${out}/scores/e2.npy" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
      --cache "${cache}" --checkpoint "${scorer}" --output "${out}/scores/e2.npy" \
      --batch-size 256 --gpu 0 >"${out}/scores/build.log" 2>&1
  fi
  if [[ ! -s "${out}/fused/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
      --cache "${cache}" --scores "${out}/scores/e2.npy" --output-dir "${out}/fused" \
      --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
      --chunk-size 256 --gpu 0 >"${out}/fused/build.log" 2>&1
  fi

  local uncertainty_dim
  uncertainty_dim=$("${PY}" -c 'import torch,sys; s=torch.load(sys.argv[1],map_location="cpu",weights_only=False); a=s["args"]; d=int(a.get("uncertainty_dim",0)); w=s["state_dict"].get("uncertainty_gate.0.weight"); print(int(w.shape[1]) if d == 0 and a.get("uncertainty_gate",False) and w is not None else d)' "${h18}")
  local uncertainty_args=()
  if (( uncertainty_dim > 0 )); then
    if [[ ! -s "${out}/uncertainty/features.npy" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/build_e2_temporal_uncertainty_20260825.py" \
        --cache "${cache}" --scores "${out}/scores/e2.npy" \
        --output "${out}/uncertainty/features.npy" --chunk-size 256 --gpu 0 \
        >"${out}/uncertainty/build.log" 2>&1
    fi
    uncertainty_args+=(--validation-uncertainty "${out}/uncertainty/features.npy")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/evaluate_frozen_h18_on_occlusion_20260822.py" \
    --validation-cache "${cache}" --validation-fused "${out}/fused/fused_poses.npy" \
    --validation-pkl "${frontend}" --checkpoint "${h18}" \
    --target-centers-pkl "${SPARSE_GT}" "${uncertainty_args[@]}" \
    --output "${out}/results/final_h18_t9.json" --frame-stride 5 \
    --batch-size 128 --gpu 0 >"${out}/results/final_h18_t9.log" 2>&1
  sha256sum "${frontend}" "${cache}" "${checkpoint}" "${scorer}" "${h18}" >"${out}/sha256.txt"
  date --iso-8601=seconds >"${out}/COMPLETED"
  echo "[stage2 ${variant} ${chain}] complete"
)

run_chain occ2 hrnet "${OCC2_GPU}" & p0=$!
run_chain occ2 resnet152 "${OCC2_GPU}" & p1=$!
run_chain occ3 hrnet "${OCC3_GPU}" & p2=$!
run_chain occ3 resnet152 "${OCC3_GPU}" & p3=$!
failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || failed=1; done
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/canonical_final_eval_COMPLETED"
"${PY}" "${AUDIT}/collect_posefusion_occ23_dense_final_table_20260824.py" \
  --output-json "${ROOT}/final_occ23_table.json" \
  --output-md "${ROOT}/final_occ23_table.md"
echo "[stage2 canonical Occ-2/Occ-3] complete"
