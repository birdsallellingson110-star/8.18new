#!/usr/bin/env bash
# HRNet counterpart of the successful ResNet Joint-Query pipeline.
# Every model/training/scorer/temporal setting is matched; only the frozen 2D
# coordinate/confidence frontend changes from ResNet-152 to HRNet-W32.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
GPU=${1:-1}
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
TEMP_FRONTEND=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
TEMP_DATASET=annot_temporal_5_5
TEMP_TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/${TEMP_DATASET}_${TYPE}
ROOT=/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet
TRAIN_ROOT=${ROOT}/generator
CACHE=${ROOT}/e2_c2/cache
HINGE=${ROOT}/e2_c2/identity_hinge
TEMP_CACHE=${ROOT}/e2_c2/temporal_cache
SCORES=${ROOT}/h18/scores
FUSED=${ROOT}/h18/fused
TEMP_OUT=${ROOT}/h18/model
TAG=HRNET_MATCHED_GBTQUERY_FULL_20E_3to1to1_seed0_20260822
RESNET_DONE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/h18_identity_hinge/COMPLETED

mkdir -p "${TYPE_DIR}" "${TEMP_TYPE_DIR}" "${TRAIN_ROOT}" "${CACHE}" "${HINGE}" "${TEMP_CACHE}" "${SCORES}" "${FUSED}" "${TEMP_OUT}"
test -s "${CFG}"; test -s "${TRAIN_FRONTEND}"; test -s "${VAL_FRONTEND}"; test -s "${TEMP_FRONTEND}"
ln -sfn "${TRAIN_FRONTEND}" "${TYPE_DIR}/h36m_train.pkl"
ln -sfn "${VAL_FRONTEND}" "${TYPE_DIR}/h36m_validation.pkl"
ln -sfn "${TEMP_FRONTEND}" "${TEMP_TYPE_DIR}/h36m_validation.pkl"

export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
set_model_env() {
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
  export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FINETUNE_LR=1e-4 RUMPL_END_EPOCH=20 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
  export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
  export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
  export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2 RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
  export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
  export RUMPL_TRAIN_SCOPE=all
}
set_model_env

# By default the matched HRNet generator can run concurrently on GPU1 while
# the ResNet downstream stages use spare capacity. Set WAIT_FOR_RESNET=1 only
# when a strictly serialized rerun is desired.
if [[ "${WAIT_FOR_RESNET:-0}" == "1" ]]; then
  while [[ ! -s "${RESNET_DONE}" ]]; do sleep 20; done
fi

if [[ ! -s "${TRAIN_ROOT}/checkpoint.txt" ]]; then
  {
    echo "comparison=matched Joint-Query pipeline; only frontend differs"
    echo "frontend=HRNet-W32 coordinates/confidence"
    echo "epochs=20 fixed_K2_epochs=8 ratio=3,1,1 lr=1e-4 seed=0"
    echo "query=global depth2 max_delta0.5; all parameters jointly trained"
    sha256sum "${CFG}" "${TRAIN_FRONTEND}" "${VAL_FRONTEND}"
  } >"${TRAIN_ROOT}/manifest.txt"
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 --train-mmpose-type "${TYPE}" \
    --test-mmpose-type "${TYPE}" --validate-on-two-datasets 0 --use-mmpose-val 1 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${TAG}" \
    >"${TRAIN_ROOT}/train.log" 2>&1
  ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${TAG}*/model_best.pth.tar" -print | sort | tail -1)
  test -s "${ckpt}"; printf '%s\n' "${ckpt}" >"${TRAIN_ROOT}/checkpoint.txt"
fi
CKPT=$(cat "${TRAIN_ROOT}/checkpoint.txt"); test -s "${CKPT}"

for views in 2 3 4; do
  eval_dir="${TRAIN_ROOT}/eval/V${views}"
  if [[ ! -s "${eval_dir}/table2.json" ]]; then
    mkdir -p "${eval_dir}"
    CUDA_VISIBLE_DEVICES="${GPU}" RUMPL_EVAL_STRICT=1 "${PY}" -u "${REPO}/run/eval_rumpl_checkpoint.py" \
      --cfg "${CFG}" --checkpoint "${CKPT}" --output-dir "${eval_dir}" --workers 8 --gpu 0 \
      --use-mmpose-val true --flip-lower-body-kp-test false --test-on-all-cameras true \
      --n-views-combinations "${views}" --model-num-views 4 --test-mmpose-type "${TYPE}" \
      >"${eval_dir}/eval.log" 2>&1
    pred=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    test -s "${pred}"
    "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
      --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
  fi
done

export_cache() {
  local split="$1" dataset="$2" output="$3"
  [[ -s "${output}" ]] && return 0
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" --dataset-name "${dataset}" --mmpose-type "${TYPE}" \
    --subset "${split}" --flip-lower-body-kp-test false --output "${output}" \
    --batch-size 256 --workers 8 --gpu 0 >"${output%.npz}.log" 2>&1
}
export_cache train annot_filtered_5_64 "${CACHE}/train_11c.npz"
export_cache validation annot_filtered_5_64 "${CACHE}/validation_11c.npz"
for split in train validation; do
  if [[ ! -s "${CACHE}/${split}_22c.npz" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${CACHE}/${split}_11c.npz" --output "${CACHE}/${split}_22c.npz" \
      >"${CACHE}/append_${split}.log" 2>&1
  fi
done
TRAIN_CACHE=${CACHE}/train_22c.npz
SPARSE_VAL_CACHE=${CACHE}/validation_22c.npz

run_hinge() {
  local seed="$1" dir="${HINGE}/seed${1}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN_CACHE}" --validation-cache "${SPARSE_VAL_CACHE}" \
    --output-dir "${dir}" --pretrain-epochs 10 --finetune-epochs 5 --batch-size 128 \
    --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --identity-hinge 0.25 --identity-v2-weight 4.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}
run_hinge 0 & e0=$!
run_hinge 1 & e1=$!
wait "${e0}" "${e1}"
if [[ ! -s "${HINGE}/calibrated_v2t04.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${SPARSE_VAL_CACHE}" --checkpoint-root "${HINGE}" \
    --output "${HINGE}/calibrated_v2t04.json" --v2-temperature 0.4 \
    --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
    >"${HINGE}/calibration.log" 2>&1
fi

export_cache validation "${TEMP_DATASET}" "${TEMP_CACHE}/validation_11c.npz"
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

if [[ ! -s "${TEMP_OUT}/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${FUSED}/train/fused_poses.npy" \
    --train-pkl "${TRAIN_FRONTEND}" --validation-cache "${VAL_CACHE}" \
    --validation-fused "${FUSED}/validation/fused_poses.npy" --validation-pkl "${TEMP_FRONTEND}" \
    --output-dir "${TEMP_OUT}" --window-length 9 --frame-stride 5 --epochs 12 --batch-size 64 \
    --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed 0 >"${TEMP_OUT}/train.log" 2>&1
fi

date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[HRNet matched Joint-Query -> identity-hinge E2-C2 -> H18] complete"
