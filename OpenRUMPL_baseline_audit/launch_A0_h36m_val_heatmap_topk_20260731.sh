#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
SCRIPT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/export_h36m_mmpose_heatmap_topk.py
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
INPUT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
IMAGES=${DATA}/images
POSE_CFG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CKPT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
OUTPUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/A0_h36m_val_heatmap_topk8
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${OUTPUT}"
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard_id % 2))
  shard="${OUTPUT}/shard${shard_id}.npz"
  dense="${OUTPUT}/shard${shard_id}.heatmaps.npy"
  log="${OUTPUT}/shard${shard_id}.log"
  if [[ -s "${shard}" && -s "${shard}.json" && -s "${dense}" ]]; then
    echo "shard ${shard_id} already complete"
    continue
  fi
  if pgrep -af "${SCRIPT}" | grep -q -- "--shard-id ${shard_id} --num-shards ${NUM_SHARDS}"; then
    echo "shard ${shard_id} already running"
    continue
  fi
  echo "start shard ${shard_id}/${NUM_SHARDS} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${SCRIPT}" \
    --input-pkl "${INPUT}" \
    --images-root "${IMAGES}" \
    --config "${POSE_CFG}" \
    --checkpoint "${POSE_CKPT}" \
    --device cuda:0 \
    --shard-id "${shard_id}" \
    --num-shards "${NUM_SHARDS}" \
    --topk 8 \
    --nms-kernel 5 \
    --dense-output "${dense}" \
    --output "${shard}" \
    >"${log}" 2>&1 &
done
wait
echo "A0 validation heatmap export complete"
