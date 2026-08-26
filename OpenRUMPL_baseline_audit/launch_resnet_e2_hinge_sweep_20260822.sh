#!/usr/bin/env bash
# Small GPU-1 sensitivity sweep while the dense ResNet temporal frontend is
# running on GPU-0.  Both jobs keep the audited ResNet H76 candidate pool and
# input fixed; only the identity-hinge strength is changed.
set -euo pipefail
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_resnet_20260821
TRAIN=${BASE}/e2_c2_cache/train_22c.npz
VAL=${BASE}/e2_c2_cache/validation_22c.npz
OUT=${BASE}/e2_c2_hinge_sweep
mkdir -p "${OUT}"
test -s "${TRAIN}" && test -s "${VAL}"

run_one() {
  local tag="$1" hinge="$2" v2w="$3"
  local dir="${OUT}/${tag}"
  [[ -s "${dir}/result.json" ]] && return 0
  mkdir -p "${dir}"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${AUDIT}" "${PY}" -u \
    "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN}" --validation-cache "${VAL}" --output-dir "${dir}" \
    --pretrain-epochs 10 --finetune-epochs 5 --batch-size 128 \
    --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --identity-hinge "${hinge}" --identity-v2-weight "${v2w}" \
    --workers 0 --seed 0 --gpu 0 >"${dir}/train.log" 2>&1
}

run_one hinge_light 0.10 2.0 & p0=$!
run_one hinge_strong 0.50 8.0 & p1=$!
wait "${p0}" "${p1}"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-E2-C2 hinge sweep] complete"
