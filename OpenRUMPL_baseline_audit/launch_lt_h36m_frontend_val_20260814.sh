#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
SCRIPT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/export_h36m_mmpose_heatmap_topk.py
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
OUTPUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/LT_H36M_frontend_val_20260814
INPUT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
IMAGES=${DATA}/images
CONFIG=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/configs/td-hm_res152_8xb32-210e_coco-384x384.py
CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
NUM_SHARDS=4
MAX_PARALLEL=${MAX_PARALLEL:-4}

mkdir -p "${OUTPUT}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

run_shard() {
  local shard_id="$1" gpu="$2"
  local stem="${OUTPUT}/shard${shard_id}"
  if [[ -s "${stem}.npz" && -s "${stem}.npz.json" && -s "${stem}.heatmaps.npy" ]]; then
    echo "shard ${shard_id} already complete"
    return
  fi
  echo "start LT-H36M ResNet-152 validation shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${SCRIPT}" \
    --input-pkl "${INPUT}" --images-root "${IMAGES}" \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --device cuda:0 --shard-id "${shard_id}" --num-shards "${NUM_SHARDS}" \
    --topk 8 --nms-kernel 5 \
    --dense-output "${stem}.heatmaps.npy" --output "${stem}.npz" \
    >"${stem}.log" 2>&1
}

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
  run_shard "${shard_id}" "$((shard_id % 2))" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo 'at least one LT-H36M validation shard failed' >&2
  exit "${status}"
fi
echo 'LT-H36M validation export complete'
