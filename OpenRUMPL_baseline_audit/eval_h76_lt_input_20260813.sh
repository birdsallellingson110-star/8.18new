#!/usr/bin/env bash
# Frozen H76 screening on official-LT 2D observations. No training or weights change.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

gpu=$1
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
MODEL_DIR=$(find /mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999 \
  -maxdepth 1 -type d -name 'H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_*' \
  -print | sort | tail -n 1)
CHECKPOINT=${MODEL_DIR}/model_best.pth.tar
CFG=$(find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.yaml' -print -quit)
TYPE=lt_alg_undistorted_annbox
OUT=/mnt/data/cjyoutput/external_fair_comparison_20260813/h76_frozen_lt_input

test -s "${CHECKPOINT}"
test -s "${CFG}"
test -s /mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${TYPE}/h36m_validation.pkl

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_EVAL_STRICT=0
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0

mkdir -p "${OUT}"
printf '%s\n' "${CHECKPOINT}" > "${OUT}/checkpoint.txt"
sha256sum "${CHECKPOINT}" "${CFG}" > "${OUT}/inputs.sha256"

cd "${REPO}"
for n_views in 2 3 4; do
  eval_dir=${OUT}/V${n_views}
  mkdir -p "${eval_dir}"
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${CHECKPOINT}" --output-dir "${eval_dir}" \
    --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-on-all-cameras true --n-views-combinations "${n_views}" \
    --model-num-views 4 --test-mmpose-type "${TYPE}" \
    > "${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" > "${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds > "${OUT}/completed.done"
