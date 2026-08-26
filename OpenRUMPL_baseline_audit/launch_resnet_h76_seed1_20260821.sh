#!/usr/bin/env bash
# Independent seed-1 replication of the ResNet H76 source model.  It uses
# exactly the same input, model flags, 8:1:1 C2 sampler and 20-E schedule as
# the seed-0 source model, and writes to a separate experiment tag.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=res152_lt_alg_undistorted_annbox
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821/h76_seed1
TAG=RES152_C2_K2HEAVY_H76_20E_LR1E4_T1_seed1_20260821

mkdir -p "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${FRONTEND}/train/h36m_train_res152.pkl" \
  && test -s "${FRONTEND}/validation/h36m_validation_res152.pkl"
for split in train validation; do
  source="${FRONTEND}/${split}/h36m_${split}_res152.pkl"
  target="${TYPE_DIR}/h36m_${split}.pkl"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || exit 2
  else
    ln -s "${source}" "${target}"
  fi
done

if [[ -s "${OUT}/checkpoint.txt" ]]; then
  echo "seed1 already complete: $(cat "${OUT}/checkpoint.txt")"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS
export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_END_EPOCH=20 RUMPL_FINETUNE_LR=1e-4 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
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

LOG=${OUT}/${TAG}.log
{
  echo "stage=ResNet152 H76 seed1 replication"
  echo "tag=${TAG} ratio=8:1:1 K2-heavy epochs=20 lr=1e-4 fixed_views=none seed=1 gpu=1"
  sha256sum "${CFG}" "${FRONTEND}/train/h36m_train_res152.pkl" "${FRONTEND}/validation/h36m_validation_res152.pkl"
} >"${LOG}"

cd "${REPO}"
"${PY}" -u run/train_rumpl.py \
  --cfg "${CFG}" --gpus 0 --workers 8 --seed 1 \
  --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
  --validate-on-two-datasets 0 --use-mmpose-val 1 \
  --apply-noise-missing 0 --missing-level 0.0 --exp-name "${TAG}" \
  >>"${LOG}" 2>&1

ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${TAG}*/model_best.pth.tar" -print | sort | tail -1)
test -s "${ckpt}"
printf '%s\n' "${ckpt}" >"${OUT}/checkpoint.txt"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-H76-seed1] complete ${ckpt}"
