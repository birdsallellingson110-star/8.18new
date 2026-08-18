#!/usr/bin/env bash
# H2: H1's single high-LR checkpoint plus confidence and robust IRLS
# triangulation candidates.  This is a solver/candidate-pool experiment; the
# RUMPL generator, HRNet input, splits, scorer and training schedule are fixed.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
SRC=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_card2_high_input_protocol_v1
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/h2_robust_input_protocol_v1
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/h2_robust_training_protocol_v1

TRAIN_SRC=${SRC}/train_h1_11c.npz
VAL_SRC=${SRC}/validation_h1_11c.npz
TRAIN_DIR=${ROOT}/train_33c
VAL_DIR=${ROOT}/validation_33c
TRAIN=${TRAIN_DIR}/validation_current_e2_33c.npz
VAL=${VAL_DIR}/validation_current_e2_33c.npz

mkdir -p "${ROOT}" "${OUT}" "${TRAIN_DIR}" "${VAL_DIR}"
test -s "${TRAIN_SRC}" && test -s "${VAL_SRC}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

{
  echo "experiment=H2_e2_card2_high_robust_33c_v1"
  echo "started=$(date --iso-8601=seconds)"
  echo "source_train=${TRAIN_SRC}"
  echo "source_validation=${VAL_SRC}"
  echo "irls_iters=3 huber_threshold_m=0.03"
} >"${OUT}/manifest.txt"

if [[ ! -s "${TRAIN}" ]]; then
  "${PY}" -u "${AUDIT}/build_current_input_e2_oracle_20260815.py" \
    --input "${TRAIN_SRC}" --output-dir "${TRAIN_DIR}" \
    --irls-iters 3 --huber-threshold-m 0.03 \
    >"${TRAIN_DIR}/build.log" 2>&1
fi
if [[ ! -s "${VAL}" ]]; then
  "${PY}" -u "${AUDIT}/build_current_input_e2_oracle_20260815.py" \
    --input "${VAL_SRC}" --output-dir "${VAL_DIR}" \
    --irls-iters 3 --huber-threshold-m 0.03 \
    >"${VAL_DIR}/build.log" 2>&1
fi
test -s "${TRAIN}" && test -s "${VAL}"

run_train() {
  local seed="$1" gpu="$2"
  local dir="${OUT}/seed${seed}"
  if [[ -s "${dir}/result.json" ]]; then
    echo "[H2] seed${seed} already complete"
    return 0
  fi
  mkdir -p "${dir}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  "${PY}" -u "${AUDIT}/train_current_e2_robust_20260815.py" \
    --train-shards "${TRAIN}" \
    --validation-cache "${VAL}" \
    --output-dir "${dir}" \
    --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}

run_train 0 0 & p0=$!
run_train 1 1 & p1=$!
wait "${p0}" "${p1}"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[H2] complete $(date --iso-8601=seconds)"
