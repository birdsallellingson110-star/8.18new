#!/usr/bin/env bash
# ResNet-152 line matched to the frozen HRNet E2-C2 -> H18-lowLR protocol.
# The old 20260817 launcher trained R0/H76 controls. This is the requested
# ResNet stage: tuned C2 source model, audited E2-C2 scorer, then H18-lowLR.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE=res152_lt_alg_undistorted_annbox
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
# The completed official LT export is in the *_gpu1 root created by the
# earlier frontend job; the non-suffixed directory only contains a failed
# preliminary launch and must not be used.
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
OUT=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999

H76_TAG=RES152_C2_K2HEAVY_H76_20E_LR1E4_T1_seed0_20260821
H76_ROOT=${OUT}/h76
H76_CKPT=${H76_ROOT}/checkpoint.txt
CACHE=${OUT}/e2_c2_cache
SCORER=${OUT}/e2_c2_scorer
SCORES=${OUT}/e2_c2_scores
FUSED=${OUT}/h18_lowlr_fused
TEMP=${OUT}/h18_lowlr

mkdir -p "${OUT}" "${H76_ROOT}" "${CACHE}" "${SCORER}" "${SCORES}" "${FUSED}" "${TEMP}" "${TYPE_DIR}"
test -s "${CFG}"
test -s "${FRONTEND}/train/h36m_train_res152.pkl"
test -s "${FRONTEND}/validation/h36m_validation_res152.pkl"

export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

for split in train validation; do
  source="${FRONTEND}/${split}/h36m_${split}_res152.pkl"
  target="${TYPE_DIR}/h36m_${split}.pkl"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      echo "mismatched ResNet dataset link ${target}" >&2; exit 2;
    }
  else
    ln -s "${source}" "${target}"
  fi
done

if [[ ! -s "${H76_CKPT}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  # The tuned C2 source-model ratio. Fixed-cardinality variables are unset so
  # they cannot silently override 8:1:1.
  unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
  export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_END_EPOCH=20 RUMPL_FINETUNE_LR=1e-4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
  export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
  export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
  export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0
  export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
  export RUMPL_TRAIN_SCOPE=all
  log=${H76_ROOT}/${H76_TAG}.log
  {
    echo "stage=ResNet152 H76 source for E2-C2"
    echo "tag=${H76_TAG} ratio=8:1:1 K2-heavy epochs=20 lr=1e-4 fixed_views=none"
    echo "frontend_train=${FRONTEND}/train/h36m_train_res152.pkl"
    echo "frontend_validation=${FRONTEND}/validation/h36m_validation_res152.pkl"
    sha256sum "${CFG}" "${FRONTEND}/train/h36m_train_res152.pkl" "${FRONTEND}/validation/h36m_validation_res152.pkl"
  } >"${log}"
  cd "${REPO}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
    --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
    --validate-on-two-datasets 0 --use-mmpose-val 1 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${H76_TAG}" \
    >>"${log}" 2>&1
  ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${H76_TAG}*/model_best.pth.tar" -print | sort | tail -1)
  test -s "${ckpt}"
  printf '%s\n' "${ckpt}" >"${H76_CKPT}"
fi

CKPT=$(cat "${H76_CKPT}")
test -s "${CKPT}"

# GPU1 is deliberately running the independent seed-1 H76 replication. Wait
# for it before launching the two-card E2 scorer, so the later parallel stage
# has exclusive access to both GPUs and no hidden memory contention.
SEED1_DONE="${OUT}/h76_seed1/COMPLETED"
if [[ ! -s "${SEED1_DONE}" ]]; then
  echo "[ResNet-E2-C2-H18] waiting for H76 seed1 replication: ${SEED1_DONE}"
  while [[ ! -s "${SEED1_DONE}" ]]; do sleep 30; done
fi

export_cache() {
  local split="$1" gpu="$2" output="$3"
  [[ -s "${output}" ]] && return 0
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
  export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
  cd "${REPO}"
  "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" --dataset-name annot_filtered_5_64 \
    --mmpose-type "${TYPE}" --subset "${split}" --flip-lower-body-kp-test false \
    --output "${output}" --batch-size 256 --workers 8 --gpu 0 \
    >"${output%.npz}.log" 2>&1
}

export_cache train 0 "${CACHE}/train_11c.npz" & p0=$!
export_cache validation 1 "${CACHE}/validation_11c.npz" & p1=$!
wait "${p0}" "${p1}"

for split in train validation; do
  input="${CACHE}/${split}_11c.npz"; output="${CACHE}/${split}_22c.npz"
  if [[ ! -s "${output}" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${input}" --output "${output}" >"${CACHE}/append_${split}.log" 2>&1
  fi
done

TRAIN_CACHE=${CACHE}/train_22c.npz
VAL_CACHE=${CACHE}/validation_22c.npz
test -s "${TRAIN_CACHE}" && test -s "${VAL_CACHE}"

run_scorer() {
  local seed="$1" gpu="$2"
  local dir="${SCORER}/seed${seed}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN_CACHE}" --validation-cache "${VAL_CACHE}" \
    --output-dir "${dir}" --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --workers 0 --seed "${seed}" --gpu 0 >"${dir}/train.log" 2>&1
}

run_scorer 0 0 & s0=$!
run_scorer 1 1 & s1=$!
wait "${s0}" "${s1}"

if [[ ! -s "${SCORER}/calibrated_v2t04.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${VAL_CACHE}" --checkpoint-root "${SCORER}" \
    --output "${SCORER}/calibrated_v2t04.json" --v2-temperature 0.4 \
    --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
    >"${SCORER}/calibration.log" 2>&1
fi

# H18-lowLR uses seed-0 E2 scores and the exact established T=9 settings.
if [[ ! -s "${SCORES}/train_e2_scores.npy" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${TRAIN_CACHE}" --checkpoint "${SCORER}/seed0/model_best.pth.tar" \
    --output "${SCORES}/train_e2_scores.npy" --batch-size 256 --gpu 0 >"${SCORES}/train.log" 2>&1
fi
if [[ ! -s "${SCORES}/validation_e2_scores.npy" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${VAL_CACHE}" --checkpoint "${SCORER}/seed0/model_best.pth.tar" \
    --output "${SCORES}/validation_e2_scores.npy" --batch-size 256 --gpu 0 >"${SCORES}/validation.log" 2>&1
fi

for split in train validation; do
  cache="${TRAIN_CACHE}"; score="${SCORES}/train_e2_scores.npy"
  [[ "${split}" == validation ]] && cache="${VAL_CACHE}" && score="${SCORES}/validation_e2_scores.npy"
  if [[ ! -s "${FUSED}/${split}/manifest.json" ]]; then
    mkdir -p "${FUSED}/${split}"
    CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
      --cache "${cache}" --scores "${score}" --output-dir "${FUSED}/${split}" \
      --chunk-size 256 --gpu 0 >"${FUSED}/${split}.log" 2>&1
  fi
done

if [[ ! -s "${TEMP}/result.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${FUSED}/train/fused_poses.npy" \
    --train-pkl "${FRONTEND}/train/h36m_train_res152.pkl" \
    --validation-cache "${VAL_CACHE}" --validation-fused "${FUSED}/validation/fused_poses.npy" \
    --validation-pkl "${FRONTEND}/validation/h36m_validation_res152.pkl" \
    --output-dir "${TEMP}" --window-length 9 --frame-stride 5 --epochs 12 --batch-size 64 \
    --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed 0 >"${TEMP}/train.log" 2>&1
fi

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-E2-C2-H18] complete ${OUT}"
