#!/usr/bin/env bash
# Train coordinate-only RUMPL controls on the GBT-style full-image HRNet cache.
#
# The script waits for the explicitly audited YOLOX shard-2 hard-case repair,
# merges all training predictions, links the same train/validation cache into
# the RUMPL dataset tree, and then launches R0 and H76 on separate GPUs.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TRAIN_RUN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train
VAL_RUN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation
TYPE=gbt_yolox_x_score001_fallback_legswap
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/rumpl_training
LOG=${OUT}/orchestrator.log

mkdir -p "${OUT}" "${TYPE_DIR}"
exec > >(tee -a "${LOG}") 2>&1
echo "[GBT-RUMPL] start $(date --iso-8601=seconds)"

while [[ ! -s "${TRAIN_RUN}/shards/shard2.pkl" && ! ( -s "${TRAIN_RUN}/shards/shard2a.pkl" && -s "${TRAIN_RUN}/shards/shard2b.pkl" ) ]]; do
  echo "[GBT-RUMPL] waiting for YOLOX shard2 (or split shard2a/shard2b); $(date --iso-8601=seconds)"
  sleep 60
done
while [[ ! -s "${TRAIN_RUN}/shards/shard2.pkl" && ! ( -s "${TRAIN_RUN}/shards/shard2a.pkl" && -s "${TRAIN_RUN}/shards/shard2b.pkl" ) ]]; do
  echo "[GBT-RUMPL] shard2 file(s) not finalized yet; $(date --iso-8601=seconds)"
  sleep 60
done

TRAIN_MERGED=${TRAIN_RUN}/merged/h36m_train.pkl
TRAIN_MANIFEST=${TRAIN_RUN}/merged/h36m_train.manifest.json
if [[ ! -s "${TRAIN_MERGED}" ]]; then
  mkdir -p "${TRAIN_RUN}/merged"
  shards=("${TRAIN_RUN}"/shards/shard*.pkl)
  if [[ ${#shards[@]} -lt 8 ]]; then
    echo "[GBT-RUMPL] expected at least 8 train shard files, got ${#shards[@]}" >&2
    exit 1
  fi
  "${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
    --input-pkl "${DATA_ROOT}/data/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --shards "${shards[@]}" --output "${TRAIN_MERGED}" \
    --manifest "${TRAIN_MANIFEST}"
fi

VAL_MERGED=${VAL_RUN}/merged/h36m_validation.pkl
if [[ ! -s "${VAL_MERGED}" ]]; then
  echo "[GBT-RUMPL] validation cache missing: ${VAL_MERGED}" >&2
  exit 1
fi

for split in train validation; do
  target=${TYPE_DIR}/h36m_${split}.pkl
  source=${TRAIN_MERGED}
  [[ "${split}" == validation ]] && source=${VAL_MERGED}
  if [[ -e "${target}" || -L "${target}" ]]; then
    existing=$(readlink -f "${target}" || true)
    expected=$(readlink -f "${source}")
    [[ "${existing}" == "${expected}" ]] || {
      echo "[GBT-RUMPL] refusing to replace ${target} -> ${existing}" >&2
      exit 1
    }
  else
    ln -s "${source}" "${target}"
  fi
done

run_one() {
  local variant=$1 gpu=$2 tri=$3 centered=$4 plucker=$5
  local tag="GBTIN_${variant}_YOLOXX_HRNet_${TYPE}_seed0_20260814"
  local root="${OUT}/${variant}"
  local log="${root}/${tag}.log"
  local done="${root}/${tag}.done"
  mkdir -p "${root}"
  if [[ -s "${done}" ]]; then
    echo "[GBT-RUMPL] ${variant} already done"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${AUDIT}"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
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
      echo "variant=${variant} gpu=${gpu} tag=${tag} start=$(date --iso-8601=seconds)"
      echo "input_type=${TYPE} train=${TRAIN_MERGED} val=${VAL_MERGED}"
      echo "variables=tri_anchor:${tri},centered:${centered},plucker:${plucker}; raw HRNet coordinates only"
      sha256sum "${CFG}" "${TRAIN_MERGED}" "${VAL_MERGED}"
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
  ) &
}

run_one R0 0 0 0 0
run_one H76 1 1 1 1
wait
echo "[GBT-RUMPL] complete $(date --iso-8601=seconds)"
