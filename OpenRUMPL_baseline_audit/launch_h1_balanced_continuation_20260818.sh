#!/usr/bin/env bash
# G1: continue the strongest V2 H1 checkpoint with balanced cardinality
# sampling. This is a single RUMPL model, not an ensemble or post-hoc fusion.
# The only experimental change from H1 is V2/V3/V4 sampling weights 1:1:1;
# input, H76 representation, camera protocol and strict evaluation are fixed.
set -euo pipefail

PY=${PY:-/home/lixiaob/cjy/rumpl_venv310/bin/python}
AUDIT=${AUDIT:-/home/lixiaob/cjy/OpenRUMPL_baseline_audit}
REPO=${REPO:-/home/lixiaob/cjy/OpenRUMPL/RUMPL}
CFG=${CFG:-/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml}
TYPE=${TYPE:-gbt_yolox_x_score001_fallback_legswap}
TRAIN=${TRAIN:-/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl}
VAL=${VAL:-/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl}
INIT=${INIT:-/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD2_HIGH_LR5E5_B2_MIXED_T1_20E_seed0_20260815_2026-08-15_18-45-38/model_best.pth.tar}
MODEL_ROOT=${MODEL_ROOT:-/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999}
OUT=${OUT:-/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h1_balanced_continuation}
TAG=${TAG:-G1_H1_BALANCED_MIXED_T1_20E_LR1E5_seed0_20260818}
VIEW_WEIGHTS=${VIEW_WEIGHTS:-1,1,1}
TYPE_DIR=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${TYPE}

mkdir -p "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${INIT}" && test -s "${TRAIN}" && test -s "${VAL}"

for split in train validation; do
  target="${TYPE_DIR}/h36m_${split}.pkl"
  source="${TRAIN}"
  [[ "${split}" == validation ]] && source="${VAL}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      echo "refusing mismatched dataset link ${target}" >&2
      exit 2
    }
  else
    ln -s "${source}" "${target}"
  fi
done

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
export RUMPL_VIEW_COUNT_WEIGHTS="${VIEW_WEIGHTS}" RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_END_EPOCH=20 RUMPL_LR_STEPS=10,15 RUMPL_FINETUNE_LR=1e-5
export RUMPL_INIT_CHECKPOINT="${INIT}"
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
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
export MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all

{
  echo "experiment=${TAG}"
  echo "started=$(date --iso-8601=seconds)"
  echo "init=${INIT}"
  echo "weights=${VIEW_WEIGHTS} end_epoch=20 lr=1e-5"
  echo "input=${TYPE}; H76 anchor+centered-Plucker; raw HRNet coordinates/confidence/rays"
  sha256sum "${CFG}" "${INIT}" "${TRAIN}" "${VAL}"
} >"${OUT}/manifest.txt"

cd "${REPO}"
"${PY}" -u run/train_rumpl.py \
  --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
  --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
  --validate-on-two-datasets 0 --use-mmpose-val 1 \
  --apply-noise-missing 0 --missing-level 0.0 --exp-name "${TAG}" \
  >"${OUT}/train.log" 2>&1

CKPT=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${TAG}*/model_best.pth.tar" -print | sort | tail -1)
test -s "${CKPT}"
printf '%s\n' "${CKPT}" >"${OUT}/checkpoint.txt"

for views in 2 3 4; do
  eval_dir="${OUT}/eval/V${views}"
  mkdir -p "${eval_dir}"
  RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${CKPT}" --output-dir "${eval_dir}" \
    --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
    --test-on-all-cameras true --n-views-combinations "${views}" \
    --model-num-views 4 --test-mmpose-type "${TYPE}" \
    >"${eval_dir}/eval.log" 2>&1
  PREDICTION=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -s "${PREDICTION}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${PREDICTION}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "${TAG} complete $(date --iso-8601=seconds)"
