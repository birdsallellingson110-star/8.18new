#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
RUMPL=/home/lixiaob/cjy/OpenRUMPL/RUMPL
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
INPUT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
TYPE=mmpose_hrnet_coco_inferencer_legswap
DEST_DIR=${DATA}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
DEST=${DEST_DIR}/h36m_validation.pkl
SHARD_DIR=${BASE}/h36m_validation_inferencer_shards
EVAL_DIR=${BASE}/H8_model_best_fullimage_inferencer_eval
CFG=${BASE}/output/multiview_amass_rumpl/multiview_rumpl_999/H8_h36m_r5scheduler_official_dualval_full128109_legmapfix_seed0_20260728_2026-07-29_00-49-44/H8_h36m_r5scheduler_official_dualval_full128109_legmapfix_seed0_20260728.yaml
CKPT=${BASE}/output/multiview_amass_rumpl/multiview_rumpl_999/H8_h36m_r5scheduler_official_dualval_full128109_legmapfix_seed0_20260728_2026-07-29_00-49-44/model_best.pth.tar
POSE_CFG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CKPT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
DET_CFG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py
DET_CKPT=/mnt/data/dataset/c2i/torch/hub/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

mkdir -p "${SHARD_DIR}" "${DEST_DIR}" "${EVAL_DIR}"

# The real-data train detections currently occupy both GPUs. H12 subsequently
# trains only on physical GPU0, so wait for all 16 train shards before using
# physical GPU1 for this independent frozen-checkpoint audit.
while true; do
  complete=$(find "${BASE}/h36m_train_mmpose_shards" -maxdepth 1 \
    -name 'shard*.pkl' -size +0c | wc -l)
  echo "$(date '+%F %T') waiting_for_train_shards=${complete}/16"
  [[ "${complete}" -eq 16 ]] && break
  sleep 60
done

export CUDA_VISIBLE_DEVICES=1
export TORCH_HOME=/mnt/data/dataset/c2i/torch
for shard in 0 1 2 3; do
  "${PY}" -u "${AUDIT}/run_h36m_mmpose_inferencer_20260729.py" \
    --input-pkl "${INPUT}" \
    --images-root "${DATA}/images" \
    --pose-config "${POSE_CFG}" \
    --pose-checkpoint "${POSE_CKPT}" \
    --det-config "${DET_CFG}" \
    --det-checkpoint "${DET_CKPT}" \
    --device cuda:0 \
    --shard-id "${shard}" \
    --num-shards 4 \
    --output "${SHARD_DIR}/shard${shard}.pkl" \
    >"${SHARD_DIR}/shard${shard}.log" 2>&1 &
done
wait

"${PY}" "${AUDIT}/merge_h36m_mmpose_hrnet_20260728.py" \
  --input-pkl "${INPUT}" \
  --shards "${SHARD_DIR}"/shard{0,1,2,3}.pkl \
  --output "${DEST}" \
  --swap-lower-body

cd "${RUMPL}"
"${PY}" -u run/eval_rumpl_checkpoint.py \
  --cfg "${CFG}" \
  --checkpoint "${CKPT}" \
  --output-dir "${EVAL_DIR}" \
  --workers 16 \
  --gpu 0 \
  --use-mmpose-val true \
  --flip-lower-body-kp-test true \
  --test-on-all-cameras true \
  --test-mmpose-type "${TYPE}" \
  >"${EVAL_DIR}/eval.log" 2>&1

DICT=$(find "${EVAL_DIR}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' | head -1)
test -n "${DICT}"
"${PY}" run/eval_h36m_table2.py \
  --dict-pkl "${DICT}" \
  --output-json "${EVAL_DIR}/table2.json" \
  >"${EVAL_DIR}/table2.log" 2>&1
cat "${EVAL_DIR}/table2.log"
