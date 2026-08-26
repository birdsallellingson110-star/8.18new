#!/usr/bin/env bash
# Export the selected deterministic GBT-style H36M-Occl HRNet coordinate cache.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
IMAGES=${DATA}/images
ROOT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/hrnet_frontend
SHARDS=${ROOT}/shards
MERGED=${ROOT}/merged/h36m_validation_hrnet_occl.pkl
MANIFEST=${ROOT}/merged/h36m_validation_hrnet_occl.manifest.json
TYPE=gbt_h36m_occl_f015_seed20260822_hrnet
TYPE_DIR=${DATA}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
DET_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_x_8xb8-300e_coco.py
DET_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
NUM_SHARDS=4

mkdir -p "${SHARDS}" "${ROOT}/logs" "${ROOT}/merged" "${TYPE_DIR}"
test -s "${GT}" && test -d "${IMAGES}"
test -s "${POSE_CONFIG}" && test -s "${POSE_CHECKPOINT}"
test -s "${DET_CONFIG}" && test -s "${DET_CHECKPOINT}"
test -s /mnt/data/cjyoutput/gbt_occlusion_stage_20260822/protocol_selection.json

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard % 2))
  output=${SHARDS}/shard${shard}.pkl
  manifest=${SHARDS}/shard${shard}.pkl.manifest.json
  if [[ -s "${output}" && -s "${manifest}" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "${PY}" -u "${AUDIT}/export_h36m_gbt_aligned_hrnet_20260814.py" \
      --input-pkl "${GT}" --images-root "${IMAGES}" \
      --pose-config "${POSE_CONFIG}" --pose-checkpoint "${POSE_CHECKPOINT}" \
      --det-config "${DET_CONFIG}" --det-checkpoint "${DET_CHECKPOINT}" \
      --score-thr 0.01 --fallback-record-box --device cuda:0 \
      --shard-id "${shard}" --num-shards "${NUM_SHARDS}" \
      --output "${output}" --manifest "${manifest}" \
      --occlusion-prob 0.1 --occlusion-square-fraction 0.15 \
      --occlusion-seed 20260822 \
      >"${ROOT}/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed )); then
  echo "one or more HRNet H36M-Occl shards failed" >&2
  exit 1
fi

shards=("${SHARDS}"/shard*.pkl)
[[ ${#shards[@]} -eq ${NUM_SHARDS} ]]
"${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
  --input-pkl "${GT}" --shards "${shards[@]}" \
  --output "${MERGED}" --manifest "${MANIFEST}" \
  >"${ROOT}/logs/merge.log" 2>&1

target=${TYPE_DIR}/h36m_validation.pkl
if [[ -e "${target}" || -L "${target}" ]]; then
  [[ "$(readlink -f "${target}")" == "$(readlink -f "${MERGED}")" ]]
else
  ln -s "${MERGED}" "${target}"
fi
sha256sum "${MERGED}" "${MANIFEST}" >"${ROOT}/sha256.txt"
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[H36M-Occl HRNet] complete: ${MERGED}"
