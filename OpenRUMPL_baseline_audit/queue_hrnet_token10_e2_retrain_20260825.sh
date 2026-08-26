#!/usr/bin/env bash
# Continue the camera-generalization branch after its frozen-candidate export.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=${HRNET_E2_ROOT:-/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_token10_generalization_20260825/canonical_e2}
CACHE=${ROOT}/cache
SCORER=${ROOT}/identity_hinge

export CUDA_VISIBLE_DEVICES="${HRNET_E2_VISIBLE_GPU:-1}"
export PYTHONPATH=${AUDIT}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "${CACHE}" "${SCORER}"

while [[ ! -s "${CACHE}/train_11c.npz" || ! -s "${CACHE}/validation_11c.npz" ]]; do
  sleep 20
done

append_candidates() {
  local split=$1
  if [[ ! -s "${CACHE}/${split}_22c.npz" ]]; then
    "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
      --input "${CACHE}/${split}_11c.npz" \
      --output "${CACHE}/${split}_22c.npz" \
      >"${CACHE}/append_${split}.log" 2>&1
  fi
}

append_candidates train & append_train=$!
append_candidates validation & append_validation=$!
wait "${append_train}" "${append_validation}"

train_seed() {
  local seed=$1
  local physical_gpu
  if [[ "${seed}" == "0" ]]; then
    physical_gpu=${HRNET_E2_SEED0_GPU:-${HRNET_E2_VISIBLE_GPU:-1}}
  else
    physical_gpu=${HRNET_E2_SEED1_GPU:-${HRNET_E2_VISIBLE_GPU:-1}}
  fi
  local out=${SCORER}/seed${seed}
  [[ -s "${out}/result.json" ]] && return 0
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${CACHE}/train_22c.npz" \
    --validation-cache "${CACHE}/validation_22c.npz" \
    --output-dir "${out}" --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 128 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --identity-hinge 0.25 --identity-v2-weight 4.0 \
    --canonical-geometry --fixed-metric-normalization --stage-heads \
    --workers 0 --seed "${seed}" --gpu 0 >"${out}/train.log" 2>&1
}

train_seed 0 & seed0=$!
train_seed 1 & seed1=$!
wait "${seed0}" "${seed1}"

CUDA_VISIBLE_DEVICES="${HRNET_E2_EVAL_GPU:-${HRNET_E2_VISIBLE_GPU:-1}}" \
"${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
  --cache "${CACHE}/validation_22c.npz" --checkpoint-root "${SCORER}" \
  --output "${SCORER}/calibrated_v2t04.json" --v2-temperature 0.4 \
  --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
  >"${SCORER}/calibration.log" 2>&1

date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[token10 E2 retrain] complete ${ROOT}"
