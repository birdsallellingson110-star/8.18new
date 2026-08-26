#!/usr/bin/env bash
# Export frozen, clean-trained HRNet and ResNet-152 2D inputs on Occ-2/Occ-3.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
ROOT=/mnt/data/cjyoutput/h36m_occ_official_20260823

POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
DET_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_x_8xb8-300e_coco.py
DET_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
LT_CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
LT_CONFIG=/home/lixiaob/cjy/reference/learnable-triangulation-official/experiments/human36m/eval/human36m_alg.yaml
NUM_SHARDS=${NUM_SHARDS:-4}
OCC2_VARIANT=${OCC2_VARIANT:-occ2}
OCC3_VARIANT=${OCC3_VARIANT:-occ3}

for required in "${GT}" "${POSE_CONFIG}" "${POSE_CHECKPOINT}" "${DET_CONFIG}" \
  "${DET_CHECKPOINT}" "${LT_CHECKPOINT}" "${LT_CONFIG}"; do
  test -s "${required}"
done

run_variant() {
  local variant=$1
  local gpu=$2
  local source_root=${ROOT}/${variant}/images
  local run=${ROOT}/${variant}/frontends
  local hrnet=${run}/hrnet
  local resnet=${run}/resnet152
  local type_hrnet=h36m_${variant}_official_hrnet_gbt_aligned
  local type_resnet=h36m_${variant}_official_res152_lt
  local type_hrnet_dir=${DATA}/data/datasets_mmpose/annot_filtered_5_64_${type_hrnet}
  local type_resnet_dir=${DATA}/data/datasets_mmpose/annot_filtered_5_64_${type_resnet}

  test -s "${ROOT}/${variant}/protocol_manifest.json"
  test -d "${source_root}"
  mkdir -p "${hrnet}/shards" "${hrnet}/logs" "${hrnet}/merged" \
    "${resnet}" "${type_hrnet_dir}" "${type_resnet_dir}"

  run_hrnet() {
    local pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      local output=${hrnet}/shards/shard${shard}.pkl
      local manifest=${output}.manifest.json
      if [[ -s "${output}" && -s "${manifest}" ]]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "${PY}" -u "${AUDIT}/export_h36m_gbt_aligned_hrnet_20260814.py" \
          --input-pkl "${GT}" --images-root "${source_root}" \
          --pose-config "${POSE_CONFIG}" --pose-checkpoint "${POSE_CHECKPOINT}" \
          --det-config "${DET_CONFIG}" --det-checkpoint "${DET_CHECKPOINT}" \
          --score-thr 0.01 --fallback-record-box --device cuda:0 \
          --shard-id "${shard}" --num-shards "${NUM_SHARDS}" \
          --output "${output}" --manifest "${manifest}" \
          >"${hrnet}/logs/shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    (( failed == 0 )) || return 1
    local completed=("${hrnet}"/shards/shard*.pkl)
    [[ ${#completed[@]} -eq ${NUM_SHARDS} ]]
    "${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
      --input-pkl "${GT}" --shards "${completed[@]}" \
      --output "${hrnet}/merged/h36m_validation.pkl" \
      --manifest "${hrnet}/merged/h36m_validation.manifest.json" \
      >"${hrnet}/logs/merge.log" 2>&1
    ln -sfn "${hrnet}/merged/h36m_validation.pkl" \
      "${type_hrnet_dir}/h36m_validation.pkl"
    sha256sum "${hrnet}/merged/h36m_validation.pkl" \
      "${hrnet}/merged/h36m_validation.manifest.json" >"${hrnet}/sha256.txt"
    date --iso-8601=seconds >"${hrnet}/COMPLETED"
  }

  run_resnet() {
    if [[ ! -s "${resnet}/h36m_validation.pkl" || ! -s "${resnet}/report.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "${PY}" -u "${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py" \
          --pkl "${GT}" --image-root "${source_root}" \
          --checkpoint "${LT_CHECKPOINT}" --config "${LT_CONFIG}" \
          --output "${resnet}/report.json" --export-only \
          --export-rumpl-pkl "${resnet}/h36m_validation.pkl" \
          --batch-size 32 --workers 8 --device cuda:0 \
          >"${resnet}/export.log" 2>&1
    fi
    ln -sfn "${resnet}/h36m_validation.pkl" \
      "${type_resnet_dir}/h36m_validation.pkl"
    sha256sum "${resnet}/h36m_validation.pkl" "${resnet}/report.json" \
      >"${resnet}/sha256.txt"
    date --iso-8601=seconds >"${resnet}/COMPLETED"
  }

  run_hrnet & local hrnet_pid=$!
  run_resnet & local resnet_pid=$!
  local failed=0
  wait "${hrnet_pid}" || failed=1
  wait "${resnet_pid}" || failed=1
  (( failed == 0 ))
  date --iso-8601=seconds >"${run}/COMPLETED"
}

run_variant "${OCC2_VARIANT}" 0 & occ2_pid=$!
run_variant "${OCC3_VARIANT}" 1 & occ3_pid=$!
failed=0
wait "${occ2_pid}" || failed=1
wait "${occ3_pid}" || failed=1
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/frontends_COMPLETED"
echo "[Human3.6M-Occluded official frontends] complete"
