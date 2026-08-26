#!/usr/bin/env bash
# Train E2-C2 with view-snap + bone-ray extra candidates.  Frozen H76 / 22c
# cache, clean H36M only, two seeds on two GPUs.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260819/e2_c2_viewsnap_bone
TRAIN=${BASE}/e2_c2_input_protocol_v2/train_c2_22c.npz
VAL=${BASE}/e2_c2_input_protocol_v2/validation_c2_22c.npz
mkdir -p "${OUT}"
test -s "${TRAIN}" && test -s "${VAL}"

run_one() {
  local seed=$1 gpu=$2
  local dir=${OUT}/seed${seed}
  mkdir -p "${dir}"
  if [[ -s "${dir}/result.json" ]]; then
    echo "[viewsnap-bone] seed${seed} already complete"
    return 0
  fi
  export CUDA_VISIBLE_DEVICES=${gpu}
  export PYTHONPATH=${AUDIT}
  "${PY}" -u "${AUDIT}/train_e2_c2_viewsnap_bone_20260819.py" \
    --train-shards "${TRAIN}" \
    --validation-cache "${VAL}" \
    --output-dir "${dir}" \
    --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}

run_one 0 0 & p0=$!
run_one 1 1 & p1=$!
wait "${p0}"
wait "${p1}"

export PYTHONPATH=${AUDIT}
"${PY}" -u "${AUDIT}/evaluate_e2_c2_viewsnap_bone_20260819.py" \
  --cache "${VAL}" \
  --train-cache "${TRAIN}" \
  --checkpoint-root "${OUT}" \
  --output "${OUT}/calibrated_v2t04.json" \
  --gpu 0

echo "[viewsnap-bone] complete $(date --iso-8601=seconds)"
