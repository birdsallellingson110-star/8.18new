#!/usr/bin/env bash
# Stage 2 (single free GPU): official LT ResNet-152 coordinate frontend,
# followed by the same RUMPL R0/H76 controls used by the HRNet line.
# GPU0 is intentionally untouched because another experiment is using it.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
SRC_DIR=${DATA_ROOT}/data/datasets/annot_filtered_5_64
TYPE=res152_lt_alg_undistorted_annbox_gpu1
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
OUT=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1
FRONTEND=${OUT}/frontend
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
LT_SCRIPT=${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py
LT_CONFIG=/home/lixiaob/cjy/reference/learnable-triangulation-official/experiments/human36m/eval/human36m_alg.yaml
LT_CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
IMAGES=${DATA_ROOT}/images
GPU=1

mkdir -p "${FRONTEND}/train" "${FRONTEND}/validation" "${TYPE_DIR}" "${OUT}/logs"
test -s "${CFG}"; test -s "${LT_SCRIPT}"; test -s "${LT_CONFIG}"; test -s "${LT_CHECKPOINT}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

export_frontend() {
  local split="$1"
  local input="${SRC_DIR}/h36m_${split}.pkl"
  local output="${FRONTEND}/${split}/h36m_${split}_res152.pkl"
  local report="${FRONTEND}/${split}/report.json"
  local log="${FRONTEND}/${split}/export.log"
  local done="${FRONTEND}/${split}/export_complete.done"
  if [[ -s "${done}" && -s "${output}" && -s "${report}" ]]; then
    echo "[RES152-GPU1] ${split} frontend already complete"
    return
  fi
  test -s "${input}"
  echo "[RES152-GPU1] exporting ${split} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${LT_SCRIPT}" \
    --pkl "${input}" --image-root "${IMAGES}" \
    --checkpoint "${LT_CHECKPOINT}" --config "${LT_CONFIG}" \
    --output "${report}" --export-only \
    --export-rumpl-pkl "${output}" --batch-size 8 --workers 8 \
    --device cuda:0 >"${log}" 2>&1
  test -s "${output}"; test -s "${report}"
  date --iso-8601=seconds >"${done}"
}

# Run sequentially on GPU1 to avoid contending with the existing GPU0 job and
# to keep peak host memory bounded while decoding the 745-MB train PKL.
export_frontend validation
export_frontend train

for split in train validation; do
  source="${FRONTEND}/${split}/h36m_${split}_res152.pkl"
  target="${TYPE_DIR}/h36m_${split}.pkl"
  test -s "${source}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    existing=$(readlink -f "${target}" || true)
    expected=$(readlink -f "${source}")
    [[ "${existing}" == "${expected}" ]] || {
      echo "refusing to replace ${target}: ${existing}" >&2
      exit 1
    }
  else
    ln -s "${source}" "${target}"
  fi
done

run_one() {
  local variant="$1" tri="$2" centered="$3" plucker="$4"
  local tag="RES152_GPU1_${variant}_LT_GBT_COORD_${TYPE}_seed0_20260817"
  local root="${OUT}/rumpl/${variant}"
  local log="${root}/${tag}.log"
  local done="${root}/${tag}.done"
  mkdir -p "${root}"
  if [[ -s "${done}" ]]; then
    echo "[RES152-GPU1] ${variant} already complete"
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export PYTHONPATH="${AUDIT}"
    export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
    export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
    export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
    export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
    export RUMPL_TRI_ANCHOR="${tri}" RUMPL_TRI_ANCHOR_REG=1e-4
    export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
    export RUMPL_ANCHOR_CENTERED_RAYS="${centered}" RUMPL_INPUT_PLUCKER="${plucker}"
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
      echo "variant=${variant} gpu=${GPU} tag=${tag} start=$(date --iso-8601=seconds)"
      echo "input_type=${TYPE}; frontend=official LT ResNet-152; source train/val=${FRONTEND}"
      echo "variables=tri_anchor:${tri},centered:${centered},plucker:${plucker}; coordinate-only"
      sha256sum "${CFG}" "${FRONTEND}/train/h36m_train_res152.pkl" "${FRONTEND}/validation/h36m_validation_res152.pkl"
    } >"${log}"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py \
      --cfg "${CFG}" --gpus 0 --workers 12 --seed 0 \
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
  )
}

# R0 is the plain coordinate control; H76 is the planned geometry control.
run_one R0 0 0 0
run_one H76 1 1 1
echo "[RES152-GPU1] complete $(date --iso-8601=seconds)"
