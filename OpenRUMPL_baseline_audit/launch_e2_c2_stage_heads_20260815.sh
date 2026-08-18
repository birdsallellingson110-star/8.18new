#!/usr/bin/env bash
# C2-based utility calibration control: share the geometry encoder but use a
# separate utility calibration head for V2/V3/V4.  The candidate pool,
# coordinates, checkpoint generator and loss schedule are unchanged from the
# audited E2-C2 protocol.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815
TRAIN=${BASE}/e2_c2_input_protocol_v2/train_c2_22c.npz
VAL=${BASE}/e2_c2_input_protocol_v2/validation_c2_22c.npz
OUT=${BASE}/e2_c2_stage_heads_protocol_v2
mkdir -p "${OUT}"
test -s "${TRAIN}" && test -s "${VAL}"

run_one() {
  local seed=$1 gpu=$2
  local dir=${OUT}/seed${seed}
  if [[ -s "${dir}/result.json" ]]; then
    echo "[stage-heads] seed${seed} already complete"
    return 0
  fi
  mkdir -p "${dir}"
  export CUDA_VISIBLE_DEVICES=${gpu}
  export PYTHONPATH=${AUDIT}
  "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${TRAIN}" \
    --validation-cache "${VAL}" \
    --output-dir "${dir}" \
    --stage-heads \
    --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}

run_one 0 0 & p0=$!
run_one 1 1 & p1=$!
wait "${p0}" "${p1}"
echo "[stage-heads] complete $(date --iso-8601=seconds)"
