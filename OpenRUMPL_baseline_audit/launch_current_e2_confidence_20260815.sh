#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_current_input
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_confidence_training

mkdir -p "${ROOT}/seed0" "${ROOT}/seed1"

run_one() {
  local gpu=$1
  local seed=$2
  local out=${ROOT}/seed${seed}
  if [[ -s "${out}/result.json" ]]; then
    echo "seed${seed} already complete; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u \
    "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${BASE}/train_h76_22c.npz" \
    --validation-cache "${BASE}/validation_h76_22c.npz" \
    --output-dir "${out}" \
    --attention-depth 2 --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --workers 0 --temperature 1.8 \
    --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --seed "${seed}" --gpu 0 \
    >"${out}/train.log" 2>&1
}

run_one 0 0 &
pid0=$!
run_one 1 1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T current-input confidence E2 completed'
