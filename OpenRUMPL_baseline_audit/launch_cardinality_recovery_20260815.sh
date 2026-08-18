#!/usr/bin/env bash
# C1/C2: recover the V3/V4 generalization lost by the K=2-only B2 run while
# retaining its or B1's useful checkpoint.  These are single-model fine-tunes,
# not stacked models.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_MERGED=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_MERGED=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/cardinality_recovery
LOG=${OUT}/orchestrator.log

B1_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/GBTCTRL_B1_CURRENTK_H76_123E_T1_seed0_20260815_2026-08-15_01-40-28/model_best.pth.tar
B2_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/GBTCTRL_B2_FIXEDK2_H76_123E_T1_seed0_20260815_2026-08-15_01-40-28/model_best.pth.tar

mkdir -p "${OUT}"
exec > >(tee -a "${LOG}") 2>&1
echo "[CARDINALITY-RECOVERY] start $(date --iso-8601=seconds)"
test -s "${B1_CKPT}" && test -s "${B2_CKPT}"
test -s "${TRAIN_MERGED}" && test -s "${VAL_MERGED}"

run_one() {
  local variant=$1 gpu=$2 init_ckpt=$3 view_weights=$4
  local tag="CARD_${variant}_H76_T1_20E_LR1e5_seed0_20260815"
  local root="${OUT}/${variant}"
  local log="${root}/${tag}.log"
  local done="${root}/${tag}.done"
  mkdir -p "${root}"
  if [[ -s "${done}" ]]; then
    echo "[CARDINALITY-RECOVERY] ${variant} already done"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${AUDIT}"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
    # Do not leave TRAIN_FIXED_NUM_VIEWS defined here: even with *_EPOCHS=0,
    # the sampler takes the fixed-cardinality branch and silently ignores
    # RUMPL_VIEW_COUNT_WEIGHTS.  These recovery runs must actually sample
    # K=2/3/4 according to the requested weights.
    unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
    export RUMPL_VIEW_COUNT_WEIGHTS="${view_weights}" RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
    export RUMPL_END_EPOCH=20 RUMPL_FINETUNE_LR=1e-5
    export RUMPL_INIT_CHECKPOINT="${init_ckpt}"
    export RUMPL_FLIP_LOWER_BODY_KP_TEST=0

    # H76 representation is unchanged; only the checkpoint and cardinality
    # sampling are manipulated in C1/C2.
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
      echo "variant=${variant} gpu=${gpu} tag=${tag} start=$(date --iso-8601=seconds)"
      echo "init=${init_ckpt} view_weights=${view_weights} fixed_epochs=0 end_epoch=20 lr=1e-5"
      echo "input_type=${TYPE} train=${TRAIN_MERGED} val=${VAL_MERGED}"
      sha256sum "${CFG}" "${init_ckpt}" "${TRAIN_MERGED}" "${VAL_MERGED}"
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

# C1: start from strong V2 B2, rehearse all cardinalities.
run_one C1_B2_TO_MIXED 0 "${B2_CKPT}" 3,1,1
# C2: start from best balanced B1, emphasize K=2 without eliminating V3/V4.
run_one C2_B1_K2HEAVY 1 "${B1_CKPT}" 8,1,1
wait
echo "[CARDINALITY-RECOVERY] complete $(date --iso-8601=seconds)"
