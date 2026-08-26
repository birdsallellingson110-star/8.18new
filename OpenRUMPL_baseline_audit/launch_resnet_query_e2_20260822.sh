#!/usr/bin/env bash
# E2-C2 scoring on the successful jointly-trained ResNet global-query model.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=res152_lt_alg_undistorted_annbox
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full
OUT=${BASE}/e2_c2
CACHE=${OUT}/cache
SCORER=${OUT}/scorer
CKPT=$(cat "${BASE}/checkpoint.txt")
mkdir -p "${CACHE}" "${SCORER}"
test -s "${CKPT}"; test -s "${CFG}"

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

TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
mkdir -p "${TYPE_DIR}"
ln -sfn "${FRONTEND}/train/h36m_train_res152.pkl" "${TYPE_DIR}/h36m_train.pkl"
ln -sfn "${FRONTEND}/validation/h36m_validation_res152.pkl" "${TYPE_DIR}/h36m_validation.pkl"

export_cache() {
  local split="$1" gpu="$2" output="$3"
  [[ -s "${output}" ]] && return 0
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" --dataset-name annot_filtered_5_64 \
    --mmpose-type "${TYPE}" --subset "${split}" --flip-lower-body-kp-test false \
    --output "${output}" --batch-size 256 --workers 8 --gpu 0 \
    >"${output%.npz}.log" 2>&1
}
export_cache train 1 "${CACHE}/train_11c.npz" & p0=$!
export_cache validation 1 "${CACHE}/validation_11c.npz" & p1=$!
wait "${p0}" "${p1}"
for split in train validation; do
  if [[ ! -s "${CACHE}/${split}_22c.npz" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${CACHE}/${split}_11c.npz" --output "${CACHE}/${split}_22c.npz" \
      >"${CACHE}/append_${split}.log" 2>&1
  fi
done

run_scorer() {
  local seed="$1" dir="${SCORER}/seed${1}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES="$((seed))" "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${CACHE}/train_22c.npz" --validation-cache "${CACHE}/validation_22c.npz" \
    --output-dir "${dir}" --pretrain-epochs 10 --finetune-epochs 5 --batch-size 256 \
    --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 --workers 0 \
    --seed "${seed}" --gpu 0 >"${dir}/train.log" 2>&1
}
run_scorer 0 & s0=$!
run_scorer 1 & s1=$!
wait "${s0}" "${s1}"
if [[ ! -s "${SCORER}/calibrated_v2t04.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${CACHE}/validation_22c.npz" --checkpoint-root "${SCORER}" \
    --output "${SCORER}/calibrated_v2t04.json" --v2-temperature 0.4 \
    --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
    >"${SCORER}/calibration.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet global-query E2] complete"
