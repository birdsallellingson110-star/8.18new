#!/usr/bin/env bash
# Two controlled robust-torso continuations, selected on S8 before formal test.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_robust_torso_20260826
BRANCH_ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/branches_20260825
CONTROL=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CAMGEN_HRNET_CANON_REPAIR_camera_ab4_token10_synth0_seed0_20260825_2026-08-25_16-44-00/final_state.pth.tar
TYPE=gbt_yolox_x_score001_fallback_legswap

mkdir -p "${ROOT}"
test -s "${CONTROL}"

run_branch() (
  set -euo pipefail
  local short_name=$1 physical_gpu=$2 dropout=$3 dropout_epochs=$4
  local repair_name=robust_torso_${short_name}_20260826
  REPAIR_NAME="${repair_name}" REPAIR_INIT="${CONTROL}" \
  REPAIR_LR=1e-6 REPAIR_EPOCHS=6 REPAIR_LR_STEPS=4 \
  REPAIR_PELVIS_PRIOR=1 REPAIR_BODY_REG=1e-2 REPAIR_ROBUST_TORSO=1 \
  REPAIR_VISIBLE_GPU="${physical_gpu}" REPAIR_VIEW_COUNT_WEIGHTS=8,1,1 \
  REPAIR_GBT_TOKEN_DROPOUT="${dropout}" \
  REPAIR_GBT_TOKEN_DROPOUT_EPOCHS="${dropout_epochs}" \
  REPAIR_SYNTHETIC_REPLACE_PROB=0 REPAIR_SKIP_FORMAL_EVAL=1 \
    bash "${AUDIT}/launch_hrnet_canonical_repair_branch_20260825.sh"
  mkdir -p "${ROOT}/${short_name}"
  cp "${BRANCH_ROOT}/${repair_name}/final_checkpoint.txt" \
    "${ROOT}/${short_name}/final_checkpoint.txt"
  cp "${BRANCH_ROOT}/${repair_name}/manifest.txt" \
    "${ROOT}/${short_name}/training_manifest.txt"
)

# The second branch checks whether token dropout helps adaptation to the new
# frame; every other training choice is matched exactly.
run_branch robust_drop0 0 0 0 & p0=$!
run_branch robust_drop10 1 0.10 2 & p1=$!
failed=0
wait "${p0}" || failed=1
wait "${p1}" || failed=1
(( failed == 0 ))

set_eval_environment() {
  export CUDA_VISIBLE_DEVICES="$1" PYTHONPATH="${AUDIT}"
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

for spec in robust_drop0:0 robust_drop10:1; do
  name=${spec%%:*}; gpu=${spec##*:}
  set_eval_environment "${gpu}"
  checkpoint=$(cat "${ROOT}/${name}/final_checkpoint.txt")
  out=${ROOT}/s8/${name}
  mkdir -p "${out}"
  if [[ ! -s "${out}/candidate_11c.npz" ]]; then
    "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
      --cfg "${CFG}" --checkpoint "${checkpoint}" \
      --dataset-name annot_filtered_5_64 --mmpose-type "${TYPE}" \
      --subset train --subjects 8 --output "${out}/candidate_11c.npz" \
      --batch-size 128 --workers 8 --gpu 0 >"${out}/export.log" 2>&1
  fi
  "${PY}" -u "${AUDIT}/evaluate_h36m_occl_direct_cache_20260822.py" \
    --cache "${out}/candidate_11c.npz" --output "${out}/direct.json" \
    >"${out}/evaluate.log" 2>&1
done

"${PY}" "${AUDIT}/select_hrnet_robust_torso_s8_20260826.py" \
  >"${ROOT}/selection_s8.log" 2>&1
selected_name=$(cat "${ROOT}/selected_name.txt")
selected=$(cat "${ROOT}/selected_checkpoint.txt")

if [[ "${selected_name}" != token10_control ]]; then
  set_eval_environment 1
  for views in 2 3 4; do
    out=${ROOT}/formal_selected/eval/V${views}
    mkdir -p "${out}"
    if [[ ! -s "${out}/table2.json" ]]; then
      cd "${REPO}"
      RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
        --cfg "${CFG}" --checkpoint "${selected}" --output-dir "${out}" \
        --workers 8 --gpu 0 --use-mmpose-val true \
        --flip-lower-body-kp-test false --test-on-all-cameras true \
        --n-views-combinations "${views}" --model-num-views 4 \
        --test-mmpose-type "${TYPE}" >"${out}/eval.log" 2>&1
      pred=$(find "${out}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${pred}"
      "${PY}" run/eval_h36m_table2.py --dict-pkl "${pred}" \
        --output-json "${out}/table2.json" >"${out}/table2.log" 2>&1
    fi
  done
else
  echo "S8 retained token10 control; formal test skipped" >"${ROOT}/FORMAL_SKIPPED"
fi

sha256sum "${CONTROL}" "${ROOT}/robust_drop0/final_checkpoint.txt" \
  "${ROOT}/robust_drop10/final_checkpoint.txt" "${ROOT}/selection_s8.json" \
  >"${ROOT}/audit.sha256"
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[HRNet robust torso] complete ${ROOT}"
