#!/usr/bin/env bash
# GBT gap control: compare the present RUMPL budget/protocol with ~300k updates.
#
# B1 keeps the current RUMPL view curriculum (fixed K=2 for 8 epochs, then
# weighted random K=2/3/4).  B2 keeps K=2 for the whole run, matching the
# published GBT training protocol.  Both use the same H76 network, input cache,
# seed, optimizer and evaluation as the active GBT-aligned controls.
#
# The script is deliberately persistent: it can be launched while the current
# R0/H76 jobs are running and will start the two controls as soon as those jobs
# finish.  It is safe to re-run; completed variants are skipped.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_MERGED=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_MERGED=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/budget_controls
LOG=${OUT}/orchestrator.log

# 2,438 batches/epoch at batch size 32. 123 epochs = 299,874 optimizer
# updates, within 0.05% of the requested 300k.  Milestones are scaled from
# the original 20-epoch [10,15] schedule to the same relative progress.
END_EPOCH=123
LR_STEPS=62,93

mkdir -p "${OUT}" "${TYPE_DIR}"
exec > >(tee -a "${LOG}") 2>&1
echo "[GBT-BUDGET] start $(date --iso-8601=seconds)"

for required in "${TRAIN_MERGED}" "${VAL_MERGED}"; do
  test -s "${required}" || {
    echo "[GBT-BUDGET] missing required cache: ${required}" >&2
    exit 1
  }
done

for split in train validation; do
  target=${TYPE_DIR}/h36m_${split}.pkl
  source=${TRAIN_MERGED}
  [[ "${split}" == validation ]] && source=${VAL_MERGED}
  if [[ -e "${target}" || -L "${target}" ]]; then
    existing=$(readlink -f "${target}" || true)
    expected=$(readlink -f "${source}")
    [[ "${existing}" == "${expected}" ]] || {
      echo "[GBT-BUDGET] refusing to replace ${target} -> ${existing}" >&2
      exit 1
    }
  else
    ln -s "${source}" "${target}"
  fi
done

# Wait for the preceding R0/H76 pair so the machine is not oversubscribed by
# four full training jobs.  If the preceding orchestrator exits without both
# done markers, fail loudly instead of silently starting an invalid comparison.
PREV=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/rumpl_training
PREV_R0=${PREV}/R0/GBTIN_R0_YOLOXX_HRNet_${TYPE}_seed0_20260814.done
PREV_H76=${PREV}/H76/GBTIN_H76_YOLOXX_HRNet_${TYPE}_seed0_20260814.done
while [[ ! -s "${PREV_R0}" || ! -s "${PREV_H76}" ]]; do
  if pgrep -f 'GBTIN_(R0|H76)_YOLOXX_HRNet_gbt_yolox_x_score001_fallback_legswap_seed0_20260814|launch_gbt_aligned_rumpl_training_20260814' >/dev/null; then
    echo "[GBT-BUDGET] waiting for preceding R0/H76 pair; $(date --iso-8601=seconds)"
    sleep 60
  else
    echo "[GBT-BUDGET] preceding jobs exited without both done markers" >&2
    exit 2
  fi
done
echo "[GBT-BUDGET] preceding pair complete; launching ${END_EPOCH} epochs (${LR_STEPS})"

run_one() {
  local variant=$1 gpu=$2 fixed_epochs=$3
  local tag="GBTCTRL_${variant}_H76_${END_EPOCH}E_T1_seed0_20260815"
  local root="${OUT}/${variant}"
  local log="${root}/${tag}.log"
  local done="${root}/${tag}.done"
  mkdir -p "${root}"
  if [[ -s "${done}" ]]; then
    echo "[GBT-BUDGET] ${variant} already done"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${AUDIT}"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
    export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS="${fixed_epochs}"
    export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
    export RUMPL_END_EPOCH="${END_EPOCH}" RUMPL_LR_STEPS="${LR_STEPS}"
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
      echo "variant=${variant} gpu=${gpu} tag=${tag} start=$(date --iso-8601=seconds)"
      echo "fixed_num_views_epochs=${fixed_epochs} end_epoch=${END_EPOCH} lr_steps=${LR_STEPS}"
      echo "input_type=${TYPE} train=${TRAIN_MERGED} val=${VAL_MERGED}"
      echo "H76=tri_anchor:centered_rays:plucker; T=1; coordinate/confidence/ray input only"
      sha256sum "${CFG}" "${TRAIN_MERGED}" "${VAL_MERGED}"
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

run_one B1_CURRENTK 0 8
run_one B2_FIXEDK2 1 "${END_EPOCH}"
wait
echo "[GBT-BUDGET] complete $(date --iso-8601=seconds)"
