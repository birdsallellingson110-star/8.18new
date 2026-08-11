#!/usr/bin/env bash
# H59: export the H58 winner on train and run the strict RUMPL evaluation.
set -euo pipefail

physical_gpu=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731

DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
A1D_TYPE=mmpose_hrnet_coco_a1d_legswap
code=${CODE_OVERRIDE:-H59}
H21=${H21_OVERRIDE:-${ROOT}/H55_H58_h21_screen/H58_balanced_views_seed0/final.pth}
TYPE=${TYPE_OVERRIDE:-mmpose_hrnet_coco_a1d_h21_h58_balanced_views_legswap}
TYPE_DIR=${DATA}/datasets_mmpose/annot_filtered_5_64_${TYPE}
BASE=${BASE_OVERRIDE:-${ROOT}/H59_h58_balanced_full}
tag=${TAG_OVERRIDE:-H59_H58balancedH21_RUMPL_workers12_seed0_20260802}
control_note=${CONTROL_NOTE_OVERRIDE:-H21 view sampling 3:1:1 to 1:1:1}
seed=${SEED_OVERRIDE:-0}
workers=${RUMPL_WORKERS:-12}
train_log=${BASE}/logs/${tag}_train.log
eval_root=${BASE}/eval/${tag}
done_file=${BASE}/completed/${tag}.done
lock_file=${BASE}/locks/${tag}.lock

mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" \
  "${BASE}/locks" "${eval_root}" "${TYPE_DIR}"
exec 9>"${lock_file}"
flock 9
if [[ -s "${done_file}" ]]; then
  echo "[${code}] skip completed"
  exit 0
fi
# shellcheck source=/dev/null
source "${AUDIT}/experiment_should_skip.sh"
if experiment_should_skip_train "${tag}" 2>/dev/null; then
  echo "[${code}] skip train (${tag} in EXPERIMENT_SKIP_REGISTRY)"
  exit 0
fi
test -s "${H21}"
test -s "${TYPE_DIR}/h36m_validation.pkl"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

if [[ ! -s "${TYPE_DIR}/h36m_train.pkl" ]]; then
  echo "[${code}] export train $(date --iso-8601=seconds)" | tee "${TYPE_DIR}/export_train.log"
  "${PY}" -u "${AUDIT}/export_h21_refined_mmpose_pkl.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --base-mmpose-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_${A1D_TYPE}/h36m_train.pkl" \
    --dense-shards "${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz \
    --h21-checkpoint "${H21}" \
    --mode a1d_h21 --a1d-checkpoint "${A1D}" --a1d-depth-samples 64 \
    --device cuda:0 --output "${TYPE_DIR}/h36m_train.pkl" \
    >>"${TYPE_DIR}/export_train.log" 2>&1
fi
test -s "${TYPE_DIR}/h36m_train.pkl"

export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
if [[ "${RUMPL_DISABLE_FIXED_VIEW_CURRICULUM:-0}" == "1" ]]; then
  unset TRAIN_FIXED_NUM_VIEWS
  export TRAIN_FIXED_NUM_VIEWS_EPOCHS=0
else
  export TRAIN_FIXED_NUM_VIEWS="${TRAIN_FIXED_NUM_VIEWS:-2}"
  export TRAIN_FIXED_NUM_VIEWS_EPOCHS="${TRAIN_FIXED_NUM_VIEWS_EPOCHS:-8}"
fi
export RUMPL_VIEW_COUNT_WEIGHTS="${RUMPL_VIEW_COUNT_WEIGHTS:-3,1,1}"
# w322 samples 2–4 views; dataset + Conv1d fusion must expose all 4 H36M cameras.
export RUMPL_N_VIEWS_TRAIN_TEST_ALL="${RUMPL_N_VIEWS_TRAIN_TEST_ALL:-4}"
export RUMPL_EVAL_STRICT="${RUMPL_EVAL_STRICT:-0}"
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_RELATIVE_VIEW_FUSION="${RUMPL_RELATIVE_VIEW_FUSION:-0}"
export RUMPL_SKELETON_VIEW_RELIABILITY="${RUMPL_SKELETON_VIEW_RELIABILITY:-0}"
export RUMPL_CONFIDENCE_VIEW_BIAS="${RUMPL_CONFIDENCE_VIEW_BIAS:-0}"
export RUMPL_GEOMETRY_VIEW_BIAS="${RUMPL_GEOMETRY_VIEW_BIAS:-0}"
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS="${RUMPL_JOINT_CONFIDENCE_VIEW_BIAS:-0}"
export RUMPL_JOINT_GEOMETRY_VIEW_BIAS="${RUMPL_JOINT_GEOMETRY_VIEW_BIAS:-0}"
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL="${RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL:-0}"
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL="${RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL:-0}"
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL="${RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL:-0}"
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL="${RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL:-0}"
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL="${RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL:-0}"
export RUMPL_TRAIN_SCOPE="${RUMPL_TRAIN_SCOPE:-all}"
export VFT_FULL_RANDOM_MASK="${VFT_FULL_RANDOM_MASK:-0}"
export RUMPL_ANCHOR_CENTERED_RAYS="${RUMPL_ANCHOR_CENTERED_RAYS:-0}"
export RUMPL_INPUT_PLUCKER="${RUMPL_INPUT_PLUCKER:-0}"
export RUMPL_INPUT_HARMONIC_L="${RUMPL_INPUT_HARMONIC_L:-0}"
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN="${RUMPL_GEOMETRY_UNCERTAINTY_TOKEN:-0}"
export GBT_GLOBAL_JV_DEPTH="${GBT_GLOBAL_JV_DEPTH:-0}"
export GBT_GLOBAL_JV_BIASED="${GBT_GLOBAL_JV_BIASED:-0}"
export GBT_GLOBAL_JV_GATED="${GBT_GLOBAL_JV_GATED:-0}"
export RUMPL_GBT_SET_DECODER="${RUMPL_GBT_SET_DECODER:-0}"
export GBT_LEARNABLE_BIAS="${GBT_LEARNABLE_BIAS:-0}"
export GBT_USE_CONF_BIAS="${GBT_USE_CONF_BIAS:-0}"
export GBT_USE_GEOM_BIAS="${GBT_USE_GEOM_BIAS:-0}"
export GBT_FUSION_GEOM="${GBT_FUSION_GEOM:-0}"
export GBT_TOKEN_DROPOUT="${GBT_TOKEN_DROPOUT:-0}"
export CAA_LAMBDA="${CAA_LAMBDA:-0}"
export DEPRO_LAMBDA="${DEPRO_LAMBDA:-0}"
export REPROJ_LAMBDA="${REPROJ_LAMBDA:-0}"
export RAY_LAMBDA="${RAY_LAMBDA:-0}"
export BONE_LAMBDA="${BONE_LAMBDA:-0}"
export MONO_W="${MONO_W:-0}"
export MONO_GT_W="${MONO_GT_W:-0}"
export MONO_MARGIN="${MONO_MARGIN:-0}"
export RUMPL_SKIP_VFT="${RUMPL_SKIP_VFT:-0}"
export RUMPL_SKIP_PFT="${RUMPL_SKIP_PFT:-0}"
export RUMPL_VFT_DEPTH="${RUMPL_VFT_DEPTH:-0}"

# shellcheck source=/dev/null
source "${AUDIT}/rumpl_stack_from_parent.sh" || true

cd "${REPO}"
checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
  -path "*${tag}*/model_best.pth.tar" -print | sort | tail -n 1)
if [[ -z "${checkpoint}" ]]; then
  stack_note="scratch"
  if [[ -n "${RUMPL_INIT_CHECKPOINT:-}" ]]; then
    stack_note="finetune_from=${RUMPL_STACK_FROM:-custom} ckpt=${RUMPL_INIT_CHECKPOINT}"
    test -s "${RUMPL_INIT_CHECKPOINT}"
  fi
  {
    echo "[${code}] train tag=${tag} $(date --iso-8601=seconds)"
    echo "[${code}] stack=${stack_note} only_variable=${control_note}"
    sha256sum "${REPO}/lib/models/multiview_rumpl.py" \
      "${REPO}/lib/models/semantic_graph_encoder.py" \
      "${REPO}/lib/models/graformer_pft.py" "${CFG}" "${H21}"
    if [[ -n "${RUMPL_INIT_CHECKPOINT:-}" ]]; then
      sha256sum "${RUMPL_INIT_CHECKPOINT}"
    fi
  } | tee "${train_log}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers "${workers}" --seed "${seed}" \
    --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
    --validate-on-two-datasets 1 --use-mmpose-val 0 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
    >>"${train_log}" 2>&1
  checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
    -path "*${tag}*/model_best.pth.tar" -print | sort | tail -n 1)
fi
test -n "${checkpoint}"
printf '%s\n' "${checkpoint}" >"${BASE}/checkpoints/${tag}.txt"

for n_views in 2 3 4; do
  eval_dir=${eval_root}/V${n_views}
  mkdir -p "${eval_dir}"
  if [[ "${n_views}" -eq 2 ]]; then test_views=(1 2)
  elif [[ "${n_views}" -eq 3 ]]; then test_views=(1 2 3)
  else test_views=(1 2 3 4)
  fi
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers "${workers}" --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-on-all-cameras true --n-views-combinations "${n_views}" \
    --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

echo "[${code}] end $(date --iso-8601=seconds)" | tee -a "${train_log}"
date --iso-8601=seconds >"${done_file}"
