#!/usr/bin/env bash
# E2-C2 scorer on the geometry-uncertainty H76 candidate pool.
set -euo pipefail
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=res152_lt_alg_undistorted_annbox
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
FRONTEND=/mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
ROOT=${BASE}/h76_geom_uncertainty
OUT=${BASE}/e2_c2_geom_uncertainty
CACHE=${OUT}/cache
SCORER=${OUT}/scorer
mkdir -p "${CACHE}" "${SCORER}"
test -s "${ROOT}/seed0/checkpoint.txt" && test -s "${FRONTEND}/train/h36m_train_res152.pkl"
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
mkdir -p "${TYPE_DIR}"
for split in train validation; do
  target="${TYPE_DIR}/h36m_${split}.pkl"; source="${FRONTEND}/${split}/h36m_${split}_res152.pkl"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || exit 2
  else
    ln -s "${source}" "${target}"
  fi
done
export PYTHONPATH="${AUDIT}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0 RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=1
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 RUMPL_TOKEN_DROPOUT=0

export_cache() {
  local split="$1" gpu="$2" output="$3"
  [[ -s "${output}" ]] && return 0
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "$(cat "${ROOT}/seed0/checkpoint.txt")" \
    --dataset-name annot_filtered_5_64 --mmpose-type "${TYPE}" --subset "${split}" \
    --flip-lower-body-kp-test false --output "${output}" --batch-size 256 --workers 8 --gpu 0 \
    >"${output%.npz}.log" 2>&1
}
export_cache train 0 "${CACHE}/train_11c.npz" & p0=$!
export_cache validation 1 "${CACHE}/validation_11c.npz" & p1=$!
wait "${p0}" "${p1}"
for split in train validation; do
  if [[ ! -s "${CACHE}/${split}_22c.npz" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${CACHE}/${split}_11c.npz" --output "${CACHE}/${split}_22c.npz" \
      >"${CACHE}/append_${split}.log" 2>&1
  fi
done
TRAIN=${CACHE}/train_22c.npz
VAL=${CACHE}/validation_22c.npz
run_scorer() {
  local seed="$1" gpu="$2"; local dir="${SCORER}/seed${seed}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN}" --validation-cache "${VAL}" --output-dir "${dir}" \
    --pretrain-epochs 10 --finetune-epochs 5 --batch-size 256 --temperature 1.8 \
    --target-temperature-mm 5.0 --oracle-weight 1.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}
run_scorer 0 0 & s0=$!
run_scorer 1 1 & s1=$!
wait "${s0}" "${s1}"
if [[ ! -s "${SCORER}/calibrated_v2t04.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${VAL}" --checkpoint-root "${SCORER}" --output "${SCORER}/calibrated_v2t04.json" \
    --v2-temperature 0.4 --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
    >"${SCORER}/calibration.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-geom-E2] complete"
