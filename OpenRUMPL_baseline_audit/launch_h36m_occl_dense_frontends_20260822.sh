#!/usr/bin/env bash
# Generate dense stride-5 H36M-Occl coordinates for frozen T=9 evaluation.
# The same deterministic mask identity and selected f=0.15 protocol are used
# for both frontends.  No model is trained or selected on these outputs.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
IMAGES=${DATA}/images
ROOT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015
SEED=20260822
FRACTION=0.15
PROB=0.1

POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
DET_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_x_8xb8-300e_coco.py
DET_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
LT_CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
LT_CONFIG=/home/lixiaob/cjy/reference/learnable-triangulation-official/experiments/human36m/eval/human36m_alg.yaml
NUM_SHARDS=${NUM_SHARDS:-8}

test -s "${GT}" && test -d "${IMAGES}"
test -s "${POSE_CONFIG}" && test -s "${POSE_CHECKPOINT}"
test -s "${DET_CONFIG}" && test -s "${DET_CHECKPOINT}"
test -s "${LT_CHECKPOINT}" && test -s "${LT_CONFIG}"

run_hrnet() {
  local run=${ROOT}/hrnet_temporal/frontend
  local shards=${run}/shards
  local merged=${run}/merged/h36m_validation_hrnet_occl_temporal.pkl
  local manifest=${run}/merged/h36m_validation_hrnet_occl_temporal.manifest.json
  local type=gbt_h36m_occl_f015_seed20260822_hrnet_temporal
  local type_dir=${DATA}/data/datasets_mmpose/annot_temporal_5_5_${type}
  mkdir -p "${shards}" "${run}/logs" "${run}/merged" "${type_dir}"
  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu=$((shard % 2))
    local output=${shards}/shard${shard}.pkl
    local shard_manifest=${shards}/shard${shard}.pkl.manifest.json
    if [[ -s "${output}" && -s "${shard_manifest}" ]]; then continue; fi
    CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      "${PY}" -u "${AUDIT}/export_h36m_gbt_aligned_hrnet_20260814.py" \
        --input-pkl "${GT}" --images-root "${IMAGES}" \
        --pose-config "${POSE_CONFIG}" --pose-checkpoint "${POSE_CHECKPOINT}" \
        --det-config "${DET_CONFIG}" --det-checkpoint "${DET_CHECKPOINT}" \
        --score-thr 0.01 --fallback-record-box --device cuda:0 \
        --shard-id "${shard}" --num-shards "${NUM_SHARDS}" \
        --output "${output}" --manifest "${shard_manifest}" \
        --occlusion-prob "${PROB}" --occlusion-square-fraction "${FRACTION}" \
        --occlusion-seed "${SEED}" >"${run}/logs/shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  (( failed == 0 )) || return 1
  local completed=("${shards}"/shard*.pkl)
  [[ ${#completed[@]} -eq ${NUM_SHARDS} ]]
  "${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
    --input-pkl "${GT}" --shards "${completed[@]}" \
    --output "${merged}" --manifest "${manifest}" >"${run}/logs/merge.log" 2>&1
  ln -sfn "${merged}" "${type_dir}/h36m_validation.pkl"
  sha256sum "${merged}" "${manifest}" >"${run}/sha256.txt"
  date --iso-8601=seconds >"${run}/COMPLETED"
}

run_resnet() {
  local run=${ROOT}/resnet_temporal/frontend
  local output=${run}/h36m_validation_res152_occl_temporal.pkl
  local report=${run}/report.json
  local type=res152_lt_alg_undistorted_annbox_occl_f015_seed20260822_temporal
  local type_dir=${DATA}/data/datasets_mmpose/annot_temporal_5_5_${type}
  mkdir -p "${run}" "${type_dir}"
  if [[ ! -s "${output}" || ! -s "${report}" ]]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      "${PY}" -u "${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py" \
        --pkl "${GT}" --image-root "${IMAGES}" --checkpoint "${LT_CHECKPOINT}" \
        --config "${LT_CONFIG}" --output "${report}" --export-only \
        --export-rumpl-pkl "${output}" --batch-size 32 --workers 8 --device cuda:0 \
        --occlusion-prob "${PROB}" --occlusion-square-fraction "${FRACTION}" \
        --occlusion-seed "${SEED}" >"${run}/export.log" 2>&1
  fi
  ln -sfn "${output}" "${type_dir}/h36m_validation.pkl"
  sha256sum "${output}" "${report}" >"${run}/sha256.txt"
  date --iso-8601=seconds >"${run}/COMPLETED"
}

run_hrnet & hrnet_pid=$!
run_resnet & resnet_pid=$!
failed=0
wait "${hrnet_pid}" || failed=1
wait "${resnet_pid}" || failed=1
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/dense_frontends_COMPLETED"
echo "[H36M-Occl dense frontends] complete"
