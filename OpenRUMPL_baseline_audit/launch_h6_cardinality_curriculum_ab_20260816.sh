#!/usr/bin/env bash
# H6: single-generator cardinality curriculum control.
#
# Both jobs start from the same B2 K=2 checkpoint and use the same 20E,
# high-LR fine-tune.  CURRICULUM changes only the sampled view-count
# distribution by epoch; FIXED_MIXED uses the established 3:1:1 distribution.
# This is a baseline-recovery experiment, not an ensemble or a view-specific
# head.  Evaluation enumerates every H36M camera combination for V2/V3/V4.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
B2=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/GBTCTRL_B2_FIXEDK2_H76_123E_T1_seed0_20260815_2026-08-15_01-40-28/model_best.pth.tar
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/h6_cardinality_curriculum_ab
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}

mkdir -p "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${B2}" && test -s "${TRAIN}" && test -s "${VAL}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

for split in train validation; do
  target="${TYPE_DIR}/h36m_${split}.pkl"
  source="${TRAIN}"
  [[ "${split}" == validation ]] && source="${VAL}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      echo "mismatched dataset link ${target}" >&2; exit 2;
    }
  else
    ln -s "${source}" "${target}"
  fi
done

run_one() {
  local variant="$1" gpu="$2" weights="$3" curriculum="$4"
  local tag="H6_${variant}_B2_H76_20E_LR5E5_T1_seed0_20260816"
  local root="${OUT}/${variant}"
  local log="${root}/${tag}.log"
  local done="${root}/${tag}.done"
  mkdir -p "${root}"
  [[ -s "${done}" ]] && { echo "[H6] ${variant} already complete"; return 0; }
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
    unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
    export RUMPL_VIEW_COUNT_WEIGHTS="${weights}" RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
    if [[ -n "${curriculum}" ]]; then
      export RUMPL_CURRICULUM_VIEW_WEIGHTS="${curriculum}"
    else
      unset RUMPL_CURRICULUM_VIEW_WEIGHTS
    fi
    export RUMPL_END_EPOCH=20 RUMPL_LR_STEPS=10,15 RUMPL_FINETUNE_LR=5e-5
    export RUMPL_INIT_CHECKPOINT="${B2}"
    export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
    export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
    export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
    export RUMPL_INPUT_HARMONIC_L=0 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
    export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
    export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
    export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
    export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
    export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
    export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
    export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
    export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
    export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
    export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0
    export MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all
    {
      echo "experiment=${tag}"
      echo "variant=${variant} gpu=${gpu} weights=${weights} curriculum=${curriculum:-none}"
      echo "started=$(date --iso-8601=seconds)"
      echo "init=${B2} type=${TYPE} train=${TRAIN} validation=${VAL} flip=false"
      sha256sum "${CFG}" "${B2}" "${TRAIN}" "${VAL}"
    } >"${log}"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py \
      --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
      --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >>"${log}" 2>&1
    ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -s "${ckpt}"
    printf '%s\n' "${ckpt}" >"${root}/checkpoint.txt"
    for views in 2 3 4; do
      eval_dir="${root}/eval/V${views}"
      mkdir -p "${eval_dir}"
      RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
        --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
        --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
        --test-on-all-cameras true --n-views-combinations "${views}" \
        --model-num-views 4 --test-mmpose-type "${TYPE}" \
        >"${eval_dir}/eval.log" 2>&1
      prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${prediction}"
      "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
        --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
    done
    date --iso-8601=seconds >"${done}"
  ) &
}

# GPU0: curriculum protects K=2, then restores K=3/K=4.
run_one CURRICULUM 0 3,1,1 '0:8,1,1;7:3,1,1;14:3,2,2'
# GPU1: same B2 initialization and budget, fixed mixed-cardinality control.
run_one FIXED_MIXED 1 3,1,1 ''
status=0
wait || status=$?
if [[ "${status}" -ne 0 ]]; then
  date --iso-8601=seconds >"${OUT}/FAILED"
  echo "[H6] failed with status=${status}" >&2
  exit "${status}"
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[H6] complete $(date --iso-8601=seconds)"
