#!/usr/bin/env bash
# Fresh E2 + selected no-warp H18 chain for the S8-selected robust HRNet model.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
SELECT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_robust_torso_20260826
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_robust_downstream_20260826
E2=${ROOT}/canonical_e2
H18=${ROOT}/canonical_h18
TYPE=gbt_yolox_x_score001_fallback_legswap

checkpoint=$(cat "${SELECT}/selected_checkpoint.txt")
selected_name=$(cat "${SELECT}/selected_name.txt")
test "${selected_name}" = robust_drop0
test -s "${checkpoint}"
mkdir -p "${E2}/cache" "${H18}/cache" "${H18}/scores"

set_generator_env() {
  export CUDA_VISIBLE_DEVICES="$1"
  export PYTHONPATH="${AUDIT}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_BODY_CANONICAL_FRAME=1 RUMPL_BODY_CANONICAL_REG=1e-2
  export RUMPL_BODY_CANONICAL_PELVIS_PRIOR=1
  export RUMPL_BODY_CANONICAL_ROBUST_TORSO=1
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
  export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
  export GBT_TOKEN_DROPOUT=0
}

export_cache() (
  set -euo pipefail
  local gpu=$1 split=$2 dataset_name=$3 output=$4
  [[ -s "${output}" ]] && exit 0
  set_generator_env "${gpu}"
  cd "${REPO}"
  "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${checkpoint}" \
    --dataset-name "${dataset_name}" --mmpose-type "${TYPE}" \
    --subset "${split}" --flip-lower-body-kp-test false \
    --output "${output}" --batch-size 256 --workers 8 --gpu 0 \
    >"${output%.npz}.log" 2>&1
)

# Keep both GPUs occupied during the expensive frozen-generator exports.
export_cache 0 train annot_filtered_5_64 "${E2}/cache/train_11c.npz" & p0=$!
export_cache 1 validation annot_filtered_5_64 "${E2}/cache/validation_11c.npz" & p1=$!
wait "${p1}"
export_cache 1 validation annot_temporal_5_5 "${H18}/cache/validation_11c.npz" & temporal=$!
wait "${p0}"

# E2 candidate construction and training exactly match the frozen token10 line.
HRNET_E2_ROOT="${E2}" HRNET_E2_VISIBLE_GPU=0 \
HRNET_E2_SEED0_GPU=0 HRNET_E2_SEED1_GPU=1 HRNET_E2_EVAL_GPU=0 \
  bash "${AUDIT}/queue_hrnet_token10_e2_retrain_20260825.sh"
wait "${temporal}"

# H18 consumes seed1, matching the established HRNet temporal line.
if [[ ! -s "${H18}/scores/train_e2_scores.npy" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${E2}/cache/train_22c.npz" \
    --checkpoint "${E2}/identity_hinge/seed1/model_best.pth.tar" \
    --output "${H18}/scores/train_e2_scores.npy" --batch-size 256 --gpu 0 \
    >"${H18}/scores/train.log" 2>&1
fi

HRNET_DOWNSTREAM_ROOT="${ROOT}" HRNET_H18_VISIBLE_GPU=1 \
HRNET_H18_NOWARP_ONLY=1 \
  bash "${AUDIT}/queue_hrnet_token10_h18_matched_20260825.sh"

{
  echo "selected_name=${selected_name}"
  echo "checkpoint=${checkpoint}"
  echo "robust_torso=1"
  echo "e2=two seeds, established identity-hinge and temperatures"
  echo "h18=continuous nowarp only; failed warp/uncertainty routes not repeated"
} >"${ROOT}/manifest.txt"
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[HRNet robust downstream] complete ${ROOT}"
