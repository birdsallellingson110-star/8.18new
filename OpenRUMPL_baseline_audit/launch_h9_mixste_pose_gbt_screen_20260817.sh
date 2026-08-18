#!/usr/bin/env bash
# H9: MixSTE STB->TTB residual on RUMPL's decoded root-relative pose.
# The input/cache/protocol is identical to H8 (GBT-aligned HRNet coordinates,
# real H36M S1/S5/S6 training, S9/S11 validation).  Only the temporal branch
# changes: it is applied after the untouched calibrated RUMPL output and its
# pelvis correction is masked to zero.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_NAME=annot_filtered_5_64
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h9_mixste_pose_residual
STEPS=${H9_STEPS:-12000}

test -s "${BASE}"
mkdir -p "${OUT}/logs"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}/lib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_VIEW_COUNT_WEIGHTS=1,0,0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_ANCHOR_CENTER_PER_JOINT=0 RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_PFT_REPEAT_LAST=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_TOKEN_DROPOUT=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0
export BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all

{
  echo "arm=H9_MIXSTE_POSE_RESIDUAL gpu=0 start=$(date --iso-8601=seconds)"
  echo "base=${BASE} train_type=${TYPE} train_dataset=${TRAIN_NAME}"
  echo "T=9 stride=5 random-K2; MixSTE STB->TTB pose residual; frozen RUMPL; steps=${STEPS}"
  sha256sum "${CFG}" "${BASE}"
} >"${OUT}/logs/H9_MIXSTE_POSE_RESIDUAL.log"

cd "${REPO}"
"${PY}" -u run/train_temporal_gbt_rumpl.py \
  --cfg "${CFG}" --base-checkpoint "${BASE}" --output-dir "${OUT}" \
  --train-mmpose-type "${TYPE}" --train-dataset-name "${TRAIN_NAME}" \
  --backbone-flavor h76 --disable-missing-keypoints \
  --window-length 9 --frame-stride 5 --num-views 2 \
  --fusion-mode mixste-pose-residual --depth 4 --heads 8 --token-dropout 0 \
  --optimizer-steps "${STEPS}" --warmup-steps 1200 \
  --micro-batch-size 64 --effective-batch-size 64 --workers 8 \
  --cache-frame-rays --cache-workers 8 \
  --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 \
  --amp-dtype bf16 --log-every 100 --save-every 2000 \
  --loss-profile mixste-original --loss-type mpjpe --loss-frame all \
  >>"${OUT}/logs/H9_MIXSTE_POSE_RESIDUAL.log" 2>&1

date --iso-8601=seconds >"${OUT}/train_complete.done"
