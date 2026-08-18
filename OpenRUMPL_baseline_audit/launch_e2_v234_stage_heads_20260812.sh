#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812
CACHE=${ROOT}/E2_V234_CandidateCache_v2
OUT=${ROOT}/E2_V234_StageHeads_20260812

mkdir -p "${OUT}/seed0" "${OUT}/seed1"

run_one() {
  local gpu=$1
  local seed=$2
  local run_dir=${OUT}/seed${seed}
  if [[ -s "${run_dir}/result.json" ]]; then
    echo "seed${seed} already has result.json; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u "${AUDIT}/train_e2_v234_universal_20260812.py" \
    --train-shards "${CACHE}/train/train_shard0of2.npz" "${CACHE}/train/train_shard1of2.npz" \
    --validation-cache "${CACHE}/validation/validation_shard0of1.npz" \
    --output-dir "${run_dir}" \
    --attention-depth 2 --stage-heads --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --workers 0 --temperature 1.8 \
    --target-temperature-mm 5.0 --oracle-weight 1.0 \
    --seed "${seed}" --gpu 0 \
    >"${run_dir}/train.log" 2>&1
}

run_one 0 0 &
pid0=$!
run_one 1 1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T E2 V234 stage-heads completed'
