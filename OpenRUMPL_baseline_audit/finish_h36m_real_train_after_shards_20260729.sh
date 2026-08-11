#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
GT_TRAIN=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_train.pkl
MMPOSE_TRAIN=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_legswap/h36m_train.pkl
SHARD_DIR=${BASE}/h36m_train_mmpose_shards
H12_CFG=${BASE}/H12_real_h36m_train.yaml
H12_NAME=H12_real_h36m_train_random_camera_subset_seed0_20260729
H12_LOG=${BASE}/H12_train_console.log

exec 9>"${BASE}/H12_real_train_finish.lock"
if ! flock -n 9; then
  echo "another H12 finish/train controller already owns the lock"
  exit 1
fi

while true; do
  complete=0
  for shard_id in $(seq 0 15); do
    if [[ -s "${SHARD_DIR}/shard${shard_id}.pkl" ]]; then
      complete=$((complete + 1))
    fi
  done
  echo "$(date '+%F %T') complete_shards=${complete}/16"
  if [[ "${complete}" -eq 16 ]]; then
    break
  fi
  sleep 60
done

shards=()
for shard_id in $(seq 0 15); do
  shards+=("${SHARD_DIR}/shard${shard_id}.pkl")
done

"${PY}" /home/lixiaob/cjy/OpenRUMPL_baseline_audit/merge_h36m_mmpose_hrnet_20260728.py \
  --input-pkl "${GT_TRAIN}" \
  --shards "${shards[@]}" \
  --output "${MMPOSE_TRAIN}" \
  --swap-lower-body

"${PY}" /home/lixiaob/cjy/OpenRUMPL_baseline_audit/audit_h36m_real_train_20260729.py \
  --dataset-pkl "${MMPOSE_TRAIN}" \
  --config "${H12_CFG}"

if pgrep -f -- "train_rumpl.py.*${H12_NAME}" >/dev/null; then
  echo "H12 is already running"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
export RUMPL_FIX_SCHEDULER_ORDER=1
cd "${REPO}"
exec "${PY}" -u run/train_rumpl.py \
  --cfg "${H12_CFG}" \
  --gpus 0 \
  --workers 16 \
  --validate-on-two-datasets 1 \
  --use-mmpose-val 0 \
  --exp-name "${H12_NAME}" \
  >"${H12_LOG}" 2>&1
