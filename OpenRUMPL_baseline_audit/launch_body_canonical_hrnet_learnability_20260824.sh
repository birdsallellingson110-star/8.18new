#!/usr/bin/env bash
# Small, matched learnability audit for the new body-canonical generator.
# This is deliberately not a paper-result run: both arms use only 2048 train
# and 1024 validation samples.  GPU 1 is left untouched for the other user's
# RayMixSTE process.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/learnability_hrnet_small
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999

mkdir -p "${ROOT}"
test -s "${CFG}"

run_arm() {
  local name=$1 canonical=$2
  local tag="CAMGEN_${name}_HRNET_2K_5E_seed0_20260824"
  local out="${ROOT}/${name}"
  mkdir -p "${out}"
  if [[ -s "${out}/COMPLETED" ]]; then
    echo "[learnability] ${name} already complete"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="${AUDIT}"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
    unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS RUMPL_INIT_CHECKPOINT
    export RUMPL_VIEW_COUNT_WEIGHTS=8,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
    export RUMPL_END_EPOCH=5 RUMPL_FINETUNE_LR=1e-4
    export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
    export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
    export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
    export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
    export RUMPL_BODY_CANONICAL_FRAME="${canonical}"
    export RUMPL_BODY_CANONICAL_REG=1e-4
    export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
    export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
    export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
    export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
    export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
    export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
    export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
    export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
    export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
    export RUMPL_GBT_QUERY_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=0
    export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
    export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_LEARNABLE_BIAS=0
    export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0
    export RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
    export RUMPL_TRAIN_SCOPE=all
    {
      echo "purpose=small learnability audit, not paper result"
      echo "arm=${name} canonical=${canonical} frontend=${TYPE}"
      echo "train_samples=2048 validation_samples=1024 epochs=5 ratio=8:1:1"
      sha256sum "${CFG}"
    } >"${out}/manifest.txt"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py \
      --cfg "${CFG}" --gpus 0 --workers 4 --seed 0 \
      --train-batch-size 32 --test-batch-size 128 \
      --train-n-samples 2048 --test-n-samples 1024 \
      --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >"${out}/train.log" 2>&1
    ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f \
      -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -s "${ckpt}"
    printf '%s\n' "${ckpt}" >"${out}/checkpoint.txt"
    date --iso-8601=seconds >"${out}/COMPLETED"
  )
}

# These two small models together fit on GPU 0.  They are run concurrently so
# the comparison sees the same machine state and finishes quickly.
run_arm world_frame 0 & p0=$!
run_arm body_canonical 1 & p1=$!
wait "${p0}" "${p1}"
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[learnability] complete ${ROOT}"
