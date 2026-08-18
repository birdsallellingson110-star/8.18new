#!/usr/bin/env bash
# Evaluate a preserved LT-input RUMPL ablation checkpoint on all H36M camera
# combinations for one view count.  This intentionally mirrors the training
# launcher environment so that the checkpoint is tested under the same model
# definition and confidence semantics.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {R0|TA|H76|LR3e5|V234Start} CHECKPOINT_TAG {2|3|4} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
checkpoint_tag=$2
n_views=$3
gpu=$4

case "${variant}" in
  R0)
    tri_anchor=0
    centered=0
    plucker=0
    ;;
  TA)
    tri_anchor=1
    centered=0
    plucker=0
    ;;
  H76|LR3e5|V234Start)
    tri_anchor=1
    centered=1
    plucker=1
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

case "${n_views}" in
  2|3|4) ;;
  *)
    echo "n_views must be 2, 3, or 4" >&2
    exit 2
    ;;
esac

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/external_fair_comparison_20260813/lt_input_rumpl_ablation
TYPE=lt_alg_undistorted_annbox
CHECKPOINT=${BASE}/epoch_checkpoints/${variant}/checkpoint_${checkpoint_tag}.pth.tar
OUTPUT=${BASE}/epoch_eval/${variant}/${checkpoint_tag}/V${n_views}
test_subset=${RUMPL_EVAL_TEST_SUBSET:-validation}
sample_groups=${RUMPL_EVAL_SAMPLE_GROUPS_PER_ACTION:-0}
sample_frames=${RUMPL_EVAL_SAMPLE_FRAMES_PER_ACTION:-0}
eval_workers=${RUMPL_EVAL_WORKERS:-8}
protocol_tag=${RUMPL_EVAL_PROTOCOL_TAG:-}
if [[ -n "${protocol_tag}" ]]; then
  OUTPUT=${BASE}/epoch_eval/${variant}/${checkpoint_tag}/V${n_views}_${protocol_tag}
fi

test -s "${CHECKPOINT}"
mkdir -p "${OUTPUT}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_TRI_ANCHOR="${tri_anchor}" RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS="${centered}" RUMPL_INPUT_PLUCKER="${plucker}"
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0 GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0
export BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all

cd "${REPO}"
RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
  --cfg "${CFG}" --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" \
  --workers "${eval_workers}" --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
  --test-on-all-cameras true --n-views-combinations "${n_views}" \
  --model-num-views 4 --test-mmpose-type "${TYPE}" \
  --test-subset "${test_subset}" \
  --sample-groups-per-action "${sample_groups}" --sample-seed 0 \
  --sample-frames-per-action "${sample_frames}" \
  --selection-manifest "${OUTPUT}/selection_manifest.json" \
  > "${OUTPUT}/eval.log" 2>&1

prediction=$(find "${OUTPUT}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
test -n "${prediction}"
"${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
  --output-json "${OUTPUT}/table2.json" > "${OUTPUT}/table2.log" 2>&1
sha256sum "${CHECKPOINT}" > "${OUTPUT}/checkpoint.sha256"
date --iso-8601=seconds > "${OUTPUT}/done.txt"
