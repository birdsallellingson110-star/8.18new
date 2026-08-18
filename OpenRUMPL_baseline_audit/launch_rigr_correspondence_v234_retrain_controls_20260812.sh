#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_V234_Retrain_20260812
CACHE=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_V234_Cache_20260812

mkdir -p "${ROOT}/no_bias_seed1" "${ROOT}/bias_seed0"

run_one() {
  local gpu=$1
  local seed=$2
  local variant=$3
  local input_variant=$4
  local out=${ROOT}/${variant}
  local cache=${CACHE}/${input_variant}
  if [[ -s "${out}/result.json" ]]; then
    echo "${variant} already has result.json; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u "${AUDIT}/train_e2_v234_universal_20260812.py" \
    --train-shards "${cache}/train/train_shard0of2.npz" "${cache}/train/train_shard1of2.npz" \
    --validation-cache "${cache}/validation/validation_shard0of1.npz" \
    --output-dir "${out}" --attention-depth 2 \
    --pretrain-epochs 10 --finetune-epochs 5 --batch-size 256 --workers 0 \
    --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --seed "${seed}" --gpu 0 >"${out}/train.log" 2>&1
}

run_one 0 1 no_bias_seed1 no_bias_seed0 &
pid0=$!
run_one 1 0 bias_seed0 bias_seed1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T correspondence V234 E2 control completed'
