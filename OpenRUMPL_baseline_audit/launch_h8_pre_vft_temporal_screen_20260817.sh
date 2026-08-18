#!/usr/bin/env bash
# H8: current GBT-aligned HRNet input + Pre-VFT temporal ray adapter.
# Two controlled arms are screened in parallel: temporal-only and low-LR
# joint fine-tuning of the retained RUMPL VFT/PFT/head.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_NAME=annot_filtered_5_64
VAL_NAME=annot_temporal_5_5
VAL_TYPE_DIR=${DATA}/data/datasets_mmpose/${VAL_NAME}_${TYPE}
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_pre_vft_temporal
STEPS=${H8_STEPS:-30000}
EVAL_LENGTHS=${H8_EVAL_LENGTHS:-"1 9"}

test -s "${BASE}"
test -s /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation_complete.done
test -s "${VAL_TYPE_DIR}/h36m_validation.pkl"
mkdir -p "${OUT}/logs"

run_one() {
  local arm=$1 gpu=$2 seed=$3 micro=$4 effective=$5 extra_train=$6
  local root=${OUT}/${arm}
  local log=${OUT}/logs/${arm}.log
  local ckpt=${root}/checkpoint_step_$(printf '%07d' "${STEPS}").pth
  mkdir -p "${root}"
  if [[ -s "${ckpt}" ]]; then
    echo "[H8] ${arm} checkpoint exists; skip training" | tee -a "${log}"
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
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
      export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
      export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
      export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
      export GBT_TOKEN_DROPOUT=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0
      export BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all
      {
        echo "arm=${arm} gpu=${gpu} seed=${seed} start=$(date --iso-8601=seconds)"
        echo "base=${BASE} train_type=${TYPE} train_dataset=${TRAIN_NAME}"
        echo "T=9 stride=5 random-K2 same subset across window; no bias/dropout/extra loss"
        echo "backbone_unfreeze=${extra_train:-none}"
        sha256sum "${CFG}" "${BASE}"
      } >"${log}"
      cd "${REPO}"
      extra=()
      if [[ -n "${extra_train}" ]]; then
        extra+=(--unfreeze-backbone --backbone-train-scope "${extra_train}" --backbone-lr-multiplier 0.1)
      fi
      "${PY}" -u run/train_temporal_gbt_rumpl.py \
        --cfg "${CFG}" --base-checkpoint "${BASE}" --output-dir "${root}" \
        --train-mmpose-type "${TYPE}" --train-dataset-name "${TRAIN_NAME}" \
        --backbone-flavor h76 --disable-missing-keypoints \
        --window-length 9 --frame-stride 5 --num-views 2 \
        --fusion-mode pre-vft-temporal --depth 2 --heads 8 --token-dropout 0 \
        --optimizer-steps "${STEPS}" --warmup-steps 3000 \
      --micro-batch-size "${micro}" --effective-batch-size "${effective}" --workers 8 \
        --cache-frame-rays --cache-workers 8 \
        --lr 0.0001 --weight-decay 0.0001 --seed "${seed}" --device cuda:0 \
        --amp-dtype bf16 --log-every 100 --save-every 5000 \
        --loss-profile rumpl --loss-type mse --loss-frame all \
        "${extra[@]}" >>"${log}" 2>&1
    )
  fi

  for t in ${EVAL_LENGTHS}; do
    for v in 2 3 4; do
      dest=${OUT}/eval/${arm}/T${t}/V${v}
      mkdir -p "${dest}"
      if [[ -s "${dest}/table2.json" ]]; then continue; fi
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${REPO}/lib" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
          --cfg "${CFG}" --base-checkpoint "${BASE}" --temporal-checkpoint "${ckpt}" \
          --backbone-flavor h76 --output-dir "${dest}" --mmpose-type "${TYPE}" \
          --dataset-name "${VAL_NAME}" --num-views "${v}" \
          --window-length "${t}" --model-window-length 9 --frame-stride 5 \
          --output-frame latest --fusion-mode pre-vft-temporal --depth 2 --heads 8 \
          --residual-scale 0.1 --flip-lower-body-kp-test false \
          --batch-size 16 --workers 6 --frame-cache "${OUT}/frame_cache/${arm}/V${v}.pt" \
          --cache-workers 16 --device cuda:0 \
          >"${dest}/eval.log" 2>&1
      pred=$(find "${dest}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${pred}"
      "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
        --output-json "${dest}/table2.json" >"${dest}/table2.log" 2>&1
    done
  done
  date --iso-8601=seconds >"${root}/screen_complete.done"
}

run_one H8_FROZEN 0 0 64 64 "" & p0=$!
run_one H8_JOINT 1 0 32 32 "vft-pft-head" & p1=$!
failed=0
wait "${p0}" || failed=1
wait "${p1}" || failed=1
(( failed == 0 ))
echo "[H8] screen complete $(date --iso-8601=seconds)"
