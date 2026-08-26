#!/usr/bin/env bash
# H21: occlusion-oriented ray temporal before VFT.
#
# H8_FROZEN already put MixSTE-style TTB on (joint, view) tracks and got
# T=9 ≈ T=1 on clean H36M.  It also turned OFF the two GBT ingredients that
# matter for occlusion: 20% token dropout and random missing keypoints.
# This run keeps H76 frozen, zero-init residual, and only adds those.
set -euo pipefail

VARIANT=${1:?usage: VARIANT PHYSICAL_GPU}
PHYSICAL_GPU=${2:?usage: VARIANT PHYSICAL_GPU}
case "${PHYSICAL_GPU}" in 0|1) ;; *) echo "gpu must be 0 or 1" >&2; exit 2 ;; esac

LOSS_PROFILE=rumpl
LOSS_TYPE=mpjpe
case "${VARIANT}" in
  h21a_missing_dropout)
    OUT_NAME=h21a_prevft_missing_dropout
    ;;
  h21b_missing_mixste_loss)
    OUT_NAME=h21b_prevft_missing_mixste_loss
    LOSS_PROFILE=mixste-original
    LOSS_TYPE=mse
    ;;
  *)
    echo "unknown variant ${VARIANT}" >&2
    exit 2
    ;;
esac

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_NAME=annot_filtered_5_64
VAL_NAME=annot_temporal_5_5
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260819
OUT=${ROOT}/${OUT_NAME}
STEPS=${H21_STEPS:-12000}
CKPT=${OUT}/checkpoint_step_$(printf '%07d' "${STEPS}").pth

test -s "${BASE}"
test -s "${DATA}/data/datasets_mmpose/${VAL_NAME}_${TYPE}/h36m_validation.pkl"
mkdir -p "${OUT}/logs" "${ROOT}/h21_frame_cache"

export PYTHONPATH="${REPO}/lib"
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_VIEW_COUNT_WEIGHTS=1,0,0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_ANCHOR_CENTER_PER_JOINT=0 RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_PFT_REPEAT_LAST=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_PER_JOINT_RESIDUAL_GATE=0 RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_TOKEN_DROPOUT=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0
export BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all

if [[ ! -s "${CKPT}" ]]; then
  {
    echo "arm=${VARIANT} gpu=${PHYSICAL_GPU} start=$(date --iso-8601=seconds)"
    echo "difference_from_H8: missing keypoints ON, GBT 20% token dropout on (J,V) tracks, lr 5e-5, frozen H76"
    echo "paper: GBT token dropout + RUMPL missing-keypoint noise; MixSTE TTB before VFT"
    sha256sum "${CFG}" "${BASE}"
  } >"${OUT}/logs/train.log"
  cd "${REPO}"
  "${PY}" -u run/train_temporal_gbt_rumpl.py \
    --cfg "${CFG}" --base-checkpoint "${BASE}" --output-dir "${OUT}" \
    --train-mmpose-type "${TYPE}" --train-dataset-name "${TRAIN_NAME}" \
    --backbone-flavor h76 \
    --window-length 9 --frame-stride 5 --num-views 2 \
    --fusion-mode pre-vft-temporal --depth 2 --heads 8 --token-dropout 0.2 \
    --optimizer-steps "${STEPS}" --warmup-steps 1200 \
    --micro-batch-size 64 --effective-batch-size 64 --workers 8 \
    --cache-frame-rays --cache-workers 8 \
    --lr 5e-5 --weight-decay 1e-4 --seed 0 --device cuda:0 \
    --amp-dtype bf16 --log-every 100 --save-every 4000 \
    --loss-profile "${LOSS_PROFILE}" --loss-type "${LOSS_TYPE}" --loss-frame all \
    >>"${OUT}/logs/train.log" 2>&1
fi

for v in 2 3 4; do
  dest=${OUT}/eval/T9/V${v}
  mkdir -p "${dest}"
  if [[ -s "${dest}/table2.json" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} PYTHONPATH="${REPO}/lib" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
      --cfg "${CFG}" --base-checkpoint "${BASE}" --temporal-checkpoint "${CKPT}" \
      --backbone-flavor h76 --output-dir "${dest}" --mmpose-type "${TYPE}" \
      --dataset-name "${VAL_NAME}" --num-views "${v}" \
      --window-length 9 --model-window-length 9 --frame-stride 5 \
      --output-frame latest --fusion-mode pre-vft-temporal --depth 2 --heads 8 \
      --residual-scale 0.1 --flip-lower-body-kp-test false \
      --batch-size 16 --workers 6 \
      --frame-cache "${ROOT}/h21_frame_cache/V${v}.pt" \
      --cache-workers 16 --device cuda:0 \
      >"${dest}/eval.log" 2>&1
  pred=$(find "${dest}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -s "${pred}"
  "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
    --output-json "${dest}/table2.json" >"${dest}/table2.log" 2>&1
done

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "${VARIANT} complete: ${OUT}"
