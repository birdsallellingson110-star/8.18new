#!/usr/bin/env bash
# Fair-input reevaluation of H15 variable-view RUMPL checkpoints.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {R0|R3} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/external_fair_comparison_20260813/raw_rumpl_h15
TYPE=mmpose_hrnet_coco_legswap

case "${variant}" in
  R0)
    pattern='H15_R0_rumpl_baseline_clean_realH36M_random2to4_seed0_*'
    conf_bias=0
    geom_bias=0
    fusion_geom=0
    learnable_bias=0
    ;;
  R3)
    pattern='H15_R3_rumpl_both_clean_realH36M_random2to4_seed0_*'
    conf_bias=1
    geom_bias=1
    fusion_geom=1
    learnable_bias=1
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

model_dir=$(find "${MODEL_ROOT}" -maxdepth 1 -type d -name "${pattern}" -print | sort | tail -n 1)
test -n "${model_dir}"
checkpoint=${model_dir}/model_best.pth.tar
cfg=$(find "${model_dir}" -maxdepth 1 -type f -name '*.yaml' -print -quit)
test -s "${checkpoint}"
test -s "${cfg}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_EVAL_STRICT=1
export RUMPL_RANDOM_VIEW_SUBSET=1

# Reconstruct only the H15 arm under test.  All later optional modules are
# explicitly disabled so an interactive shell cannot leak experiment flags.
export GBT_LEARNABLE_BIAS="${learnable_bias}"
export GBT_USE_CONF_BIAS="${conf_bias}"
export GBT_USE_GEOM_BIAS="${geom_bias}"
export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=1.0
export GBT_FUSION_GEOM="${fusion_geom}"
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_TOKEN_DROPOUT=0
export RUMPL_TRI_ANCHOR=0 RUMPL_KPA=0 RUMPL_ANCHOR_CENTERED_RAYS=0
export RUMPL_INPUT_PLUCKER=0 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_GBT_SET_DECODER=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS

mkdir -p "${OUT}/${variant}"
printf '%s\n' "${checkpoint}" > "${OUT}/${variant}/checkpoint.txt"
sha256sum "${checkpoint}" "${cfg}" > "${OUT}/${variant}/inputs.sha256"

cd "${REPO}"
for n_views in 2 3 4; do
  eval_dir=${OUT}/${variant}/V${n_views}
  mkdir -p "${eval_dir}"
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${cfg}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-on-all-cameras true --n-views-combinations "${n_views}" \
    --model-num-views 2 \
    --test-mmpose-type "${TYPE}" > "${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" > "${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds > "${OUT}/${variant}/completed.done"
