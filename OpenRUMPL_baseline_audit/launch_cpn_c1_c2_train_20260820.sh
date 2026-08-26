#!/usr/bin/env bash
# Same-model CPN controls.  C1/C2 differ only in the converted 2-D pkl
# (coordinates are identical; C2 additionally supplies the official score).
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
CPN_ROOT=/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict
OUT="${CPN_OUT:-/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict/trained_same_model}"

mkdir -p "${OUT}"
test -s "${CFG}"

link_variant() {
  local variant="$1"
  local type="mtf_cpn_native_${variant,,}"
  local dir="${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${type}"
  mkdir -p "${dir}"
  for split in train validation; do
    local source="${CPN_ROOT}/h36m_${split}_${variant}.pkl"
    local target="${dir}/h36m_${split}.pkl"
    test -s "${source}"
    if [[ -e "${target}" || -L "${target}" ]]; then
      [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
        echo "mismatched CPN link: ${target}" >&2; exit 2;
      }
    else
      ln -s "${source}" "${target}"
    fi
  done
}

run_variant() {
  local variant="$1" gpu="$2"
  local type="mtf_cpn_native_${variant,,}"
  local tag="CPNTRAIN_${variant}_H76_T1_20E_K2HEAVY_seed0_20260820"
  local root="${OUT}/${variant}"
  local log="${root}/${tag}.log"
  local done="${root}/${tag}.done"
  mkdir -p "${root}"
  [[ -s "${done}" ]] && return 0
  link_variant "${variant}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export PYTHONPATH="${AUDIT}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  # Exact H76 control: no architecture/module change, only the annotation type.
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
  export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
  export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
  export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0
  export MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all
  {
    echo "variant=${variant} type=${type} gpu=${gpu} tag=${tag} start=$(date --iso-8601=seconds)"
    echo "model=unchanged_H76_architecture; only CPN pkl type changes"
    sha256sum "${CFG}" "${CPN_ROOT}/h36m_train_${variant}.pkl" "${CPN_ROOT}/h36m_validation_${variant}.pkl"
  } >"${log}"
  cd "${REPO}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 12 --seed 0 \
    --train-mmpose-type "${type}" --test-mmpose-type "${type}" \
    --validate-on-two-datasets 0 --use-mmpose-val 1 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
    >>"${log}" 2>&1
  local ckpt
  ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
  test -s "${ckpt}"
  printf '%s\n' "${ckpt}" >"${root}/checkpoint.txt"
  for views in 2 3 4; do
    local eval_dir="${root}/eval/V${views}"
    mkdir -p "${eval_dir}"
    RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
      --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
      --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
      --test-on-all-cameras true --n-views-combinations "${views}" \
      --model-num-views 4 --test-mmpose-type "${type}" \
      >"${eval_dir}/eval.log" 2>&1
    local prediction
    prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    test -s "${prediction}"
    "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
      --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
  done
  date --iso-8601=seconds >"${done}"
}

gpu_c1="${CPN_GPU_C1:-0}"
gpu_c2="${CPN_GPU_C2:-1}"
if [[ "${CPN_ONLY_VARIANT:-}" == "C1" ]]; then
  run_variant C1 "${gpu_c1}"
elif [[ "${CPN_ONLY_VARIANT:-}" == "C2" ]]; then
  run_variant C2 "${gpu_c2}"
else
  run_variant C1 "${gpu_c1}" & p0=$!
  run_variant C2 "${gpu_c2}" & p1=$!
  wait "${p0}" "${p1}"
fi
