#!/usr/bin/env bash
# ResNet-input E2-C2 identity-hinge control.  Candidate pool and input are
# identical to the main scorer; only the previously registered identity hinge
# is changed, so its result is a clean module ablation.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
TRAIN=${BASE}/e2_c2_cache/train_22c.npz
VAL=${BASE}/e2_c2_cache/validation_22c.npz
OUT=${BASE}/e2_c2_scorer_hinge
mkdir -p "${OUT}"
test -s "${TRAIN}" && test -s "${VAL}"

run_one() {
  local seed="$1" gpu="$2"
  local dir="${OUT}/seed${seed}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${AUDIT}" \
    "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN}" --validation-cache "${VAL}" \
    --output-dir "${dir}" --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 128 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --identity-hinge 0.25 --identity-v2-weight 4.0 \
    --workers 0 --seed "${seed}" --gpu 0 >"${dir}/train.log" 2>&1
}

run_one 0 0 & p0=$!
run_one 1 1 & p1=$!
wait "${p0}" "${p1}"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${AUDIT}" \
  "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
  --cache "${VAL}" --checkpoint-root "${OUT}" \
  --output "${OUT}/calibrated_v2t04.json" --v2-temperature 0.4 \
  --v3-temperature 1.8 --v4-temperature 1.8 --batch-size 1024 --gpu 0 \
  >"${OUT}/calibration.log" 2>&1
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-E2-C2-hinge] complete"
