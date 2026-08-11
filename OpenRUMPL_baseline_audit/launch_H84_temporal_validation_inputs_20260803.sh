#!/usr/bin/env bash
# Build dense stride-5 H36M validation inputs for fair temporal evaluation.
set -euo pipefail

physical_gpu=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
GT=${DATA}/datasets/annot_temporal_5_5/h36m_validation.pkl
IMAGES=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/images
OUT=${ROOT}/H84_temporal_stride5_validation_inputs
SHARDS=${OUT}/hrnet_dense_topk8
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
TYPE_DIR=${DATA}/datasets_mmpose/annot_temporal_5_5_${TYPE}
BASE=${OUT}/h36m_validation_hrnet_coco_legswap.pkl
FINAL=${TYPE_DIR}/h36m_validation.pkl
CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
H21=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
num_shards=8

mkdir -p "${OUT}/logs" "${SHARDS}" "${TYPE_DIR}"
exec 9>"${OUT}/pipeline.lock"
flock 9
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[H84] start $(date --iso-8601=seconds) physical_gpu=${physical_gpu}" | tee -a "${OUT}/pipeline.log"
sha256sum "${GT}" "${AUDIT}/export_h36m_mmpose_heatmap_topk.py" \
  "${AUDIT}/merge_h36m_mmpose_hrnet_20260728.py" \
  "${AUDIT}/export_h21_refined_mmpose_pkl.py" >>"${OUT}/pipeline.log"

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  sparse=${SHARDS}/shard${shard}.npz
  dense=${SHARDS}/shard${shard}.heatmaps.npy
  if [[ -s "${sparse}" && -s "${dense}" ]]; then
    echo "[H84] shard ${shard} already complete" | tee -a "${OUT}/pipeline.log"
    continue
  fi
  "${PY}" -u "${AUDIT}/export_h36m_mmpose_heatmap_topk.py" \
    --input-pkl "${GT}" --images-root "${IMAGES}" \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --device cuda:0 \
    --shard-id "${shard}" --num-shards "${num_shards}" --topk 8 \
    --output "${sparse}" --dense-output "${dense}" \
    >"${OUT}/logs/hrnet_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

shard_npz=()
for shard in $(seq 0 $((num_shards - 1))); do
  test -s "${SHARDS}/shard${shard}.npz"
  test -s "${SHARDS}/shard${shard}.heatmaps.npy"
  shard_npz+=("${SHARDS}/shard${shard}.npz")
done

if [[ ! -s "${BASE}" ]]; then
  "${PY}" -u "${AUDIT}/merge_h36m_mmpose_hrnet_20260728.py" \
    --input-pkl "${GT}" --shards "${shard_npz[@]}" \
    --swap-lower-body --output "${BASE}" >"${OUT}/logs/merge_base.log" 2>&1
fi

if [[ ! -s "${FINAL}" ]]; then
  "${PY}" -u "${AUDIT}/export_h21_refined_mmpose_pkl.py" \
    --input-pkl "${GT}" --base-mmpose-pkl "${BASE}" \
    --dense-shards "${shard_npz[@]}" --h21-checkpoint "${H21}" \
    --mode a1d_h21 --a1d-checkpoint "${A1D}" --a1d-depth-samples 64 \
    --device cuda:0 --log-every 500 \
    --output "${FINAL}" >"${OUT}/logs/export_a1d_h21.log" 2>&1
fi

date --iso-8601=seconds >"${OUT}/completed.done"
echo "[H84] complete $(date --iso-8601=seconds) final=${FINAL}" | tee -a "${OUT}/pipeline.log"
