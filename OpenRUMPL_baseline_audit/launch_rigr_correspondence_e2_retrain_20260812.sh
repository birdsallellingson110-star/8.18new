#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Retrain_20260812
EXPORT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Export_20260812

mkdir -p "${ROOT}/no_bias_seed0" "${ROOT}/bias_seed1"

run_one() {
  local gpu=$1
  local seed=$2
  local variant=$3
  local input=${EXPORT}/${variant}
  local out=${ROOT}/${variant}
  if [[ -s "${out}/result.json" ]]; then
    echo "${variant} already has result.json; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u "${AUDIT}/train_h76_learned_candidate_e2_20260814.py" \
    --train-shards "${input}/train_rigr_e2_shard0of2.npz" "${input}/train_rigr_e2_shard1of2.npz" \
    --validation-cache "${input}/validation_rigr_e2.npz" \
    --output-dir "${out}" --attention-depth 2 \
    --pretrain-epochs 10 --finetune-epochs 5 --temperature 1.8 \
    --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --batch-size 256 --workers 0 --seed "${seed}" --gpu 0 \
    >"${out}/train.log" 2>&1
}

run_one 0 0 no_bias_seed0 &
pid0=$!
run_one 1 1 bias_seed1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T RIGR correspondence E2 retraining completed'
