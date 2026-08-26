#!/usr/bin/env bash
# Complete the successful ResNet global Joint-Query model with the registered
# best E2-C2 identity-preserving scorer and the frozen H18-lowLR T=9 module.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GPU=${1:-1}
TYPE=res152_lt_alg_undistorted_annbox
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full
STANDARD=${BASE}/e2_c2
CACHE=${STANDARD}/cache
HINGE=${BASE}/e2_c2_identity_hinge
TEMP_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/frontend_temporal_v2_gtinput/validation/h36m_validation_res152_temporal.pkl
TRAIN_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/train/h36m_train_res152.pkl
TEMP_DATASET=annot_filtered_5_64_res152_query_temporal_v2_gtinput
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/${TEMP_DATASET}_${TYPE}
TEMP_CACHE=${BASE}/temporal_cache_v2_gtinput
SCORES=${BASE}/identity_hinge_temporal_scores_v2_gtinput
FUSED=${BASE}/identity_hinge_temporal_fused_v2_gtinput
OUT=${BASE}/h18_identity_hinge_v2_gtinput

mkdir -p "${HINGE}" "${TYPE_DIR}" "${TEMP_CACHE}" "${SCORES}" "${FUSED}" "${OUT}"
test -s "${CFG}"; test -s "${BASE}/checkpoint.txt"; test -s "${TRAIN_FRONTEND}"; test -s "${TEMP_FRONTEND}"
ln -sfn "${TEMP_FRONTEND}" "${TYPE_DIR}/h36m_validation.pkl"

export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 RUMPL_TOKEN_DROPOUT=0

# Keep the standard E2 result as the scorer ablation, but use the previously
# registered identity-preserving variant for the final pipeline.
while [[ ! -s "${STANDARD}/COMPLETED" ]]; do sleep 20; done
TRAIN_CACHE=${CACHE}/train_22c.npz
SPARSE_VAL_CACHE=${CACHE}/validation_22c.npz
test -s "${TRAIN_CACHE}"; test -s "${SPARSE_VAL_CACHE}"

run_hinge() {
  local seed="$1" gpu="$2" dir="${HINGE}/seed${1}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN_CACHE}" --validation-cache "${SPARSE_VAL_CACHE}" \
    --output-dir "${dir}" --pretrain-epochs 10 --finetune-epochs 5 --batch-size 128 \
    --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --identity-hinge 0.25 --identity-v2-weight 4.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}
run_hinge 0 0 & h0=$!
run_hinge 1 1 & h1=$!
wait "${h0}" "${h1}"

if [[ ! -s "${HINGE}/calibrated_v2t04.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${SPARSE_VAL_CACHE}" --checkpoint-root "${HINGE}" \
    --output "${HINGE}/calibrated_v2t04.json" --v2-temperature 0.4 \
    --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
    >"${HINGE}/calibration.log" 2>&1
fi

# Dense validation hypotheses are model-specific and cannot reuse the old H76 cache.
if [[ ! -s "${TEMP_CACHE}/validation_11c.npz" ]]; then
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "$(cat "${BASE}/checkpoint.txt")" \
    --dataset-name "${TEMP_DATASET}" --mmpose-type "${TYPE}" --subset validation \
    --flip-lower-body-kp-test false --output "${TEMP_CACHE}/validation_11c.npz" \
    --batch-size 256 --workers 8 --gpu 0 >"${TEMP_CACHE}/validation_11c.log" 2>&1
fi
if [[ ! -s "${TEMP_CACHE}/validation_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${TEMP_CACHE}/validation_11c.npz" --output "${TEMP_CACHE}/validation_22c.npz" \
    >"${TEMP_CACHE}/append_validation.log" 2>&1
fi
VAL_CACHE=${TEMP_CACHE}/validation_22c.npz

for split in train validation; do
  cache="${TRAIN_CACHE}"; score="${SCORES}/train_e2_scores.npy"
  [[ "${split}" == validation ]] && cache="${VAL_CACHE}" && score="${SCORES}/validation_e2_scores.npy"
  if [[ ! -s "${score}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
      --cache "${cache}" --checkpoint "${HINGE}/seed0/model_best.pth.tar" \
      --output "${score}" --batch-size 256 --gpu 0 >"${SCORES}/${split}.log" 2>&1
  fi
  if [[ ! -s "${FUSED}/${split}/manifest.json" ]]; then
    mkdir -p "${FUSED}/${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
      --cache "${cache}" --scores "${score}" --output-dir "${FUSED}/${split}" \
      --temperature-v2 0.4 --temperature-v3 1.8 --temperature-v4 1.8 \
      --chunk-size 256 --gpu 0 >"${FUSED}/${split}.log" 2>&1
  fi
done

if [[ ! -s "${OUT}/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${FUSED}/train/fused_poses.npy" \
    --train-pkl "${TRAIN_FRONTEND}" --validation-cache "${VAL_CACHE}" \
    --validation-fused "${FUSED}/validation/fused_poses.npy" --validation-pkl "${TEMP_FRONTEND}" \
    --output-dir "${OUT}" --window-length 9 --frame-stride 5 --epochs 12 --batch-size 64 \
    --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed 0 >"${OUT}/train.log" 2>&1
fi

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet Joint-Query -> identity-hinge E2-C2 -> H18] complete"
