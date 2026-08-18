#!/usr/bin/env bash
# GBT-style two-view specialist: train the same E2-C2 utility model and
# candidate pool, but expose only the six V2 tasks during optimization and
# checkpoint selection.  This is a diagnostic for the current V2 bottleneck;
# it must not be merged with the V3/V4 E2-C2 numbers unless it passes all
# cardinalities in a separate evaluation.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815
TRAIN=${BASE}/e2_c2_input_protocol_v2/train_c2_22c.npz
VAL=${BASE}/e2_c2_input_protocol_v2/validation_c2_22c.npz
OUT=${BASE}/e2_c2_v2_specialist_protocol_v1
mkdir -p "${OUT}"
test -s "${TRAIN}" && test -s "${VAL}"

run_one() {
  local seed="$1" gpu="$2"
  local dir=${OUT}/seed${seed}
  if [[ -s "${dir}/result.json" ]]; then
    echo "[v2-specialist] seed${seed} already complete"
    return 0
  fi
  mkdir -p "${dir}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export PYTHONPATH="${AUDIT}"
  "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN}" \
    --validation-cache "${VAL}" \
    --output-dir "${dir}" \
    --task-cardinalities 2 \
    --pretrain-epochs 20 --finetune-epochs 10 \
    --batch-size 256 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}

run_one 0 0 & p0=$!
run_one 1 1 & p1=$!
wait "${p0}" "${p1}"
echo "[v2-specialist] complete $(date --iso-8601=seconds)"
