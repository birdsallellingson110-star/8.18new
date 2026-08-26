#!/usr/bin/env bash
# Export frozen HRNet-W32 and official LT ResNet-152 coordinates on dense Occ-2/3.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
ROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824

POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
DET_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_x_8xb8-300e_coco.py
DET_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
LT_CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
LT_CONFIG=/home/lixiaob/cjy/reference/learnable-triangulation-official/experiments/human36m/eval/human36m_alg.yaml
NUM_SHARDS=${NUM_SHARDS:-4}
OCC2_GPU=${OCC2_GPU:-0}
OCC3_GPU=${OCC3_GPU:-1}
SERIAL_VARIANTS=${SERIAL_VARIANTS:-0}

for required in "${GT}" "${POSE_CONFIG}" "${POSE_CHECKPOINT}" "${DET_CONFIG}" \
  "${DET_CHECKPOINT}" "${LT_CHECKPOINT}" "${LT_CONFIG}"; do
  test -s "${required}"
done
while [[ ! -s "${ROOT}/generation_COMPLETED" ]]; do sleep 20; done

# Keep the scored 2021 center frames byte-identical to the sparse benchmark
# whose official Algebraic control matches the published Occ-2/Occ-3 table.
SPARSE_GT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
SPARSE_ROOT=/mnt/data/cjyoutput/h36m_occ_official_20260823
for views in 2 3; do
  dense=${ROOT}/occ${views}
  sparse=${SPARSE_ROOT}/calib_c2_s020_050_occ${views}
  if [[ ! -s "${dense}/sparse_centers_OVERLAID" ]]; then
    "${PY}" -u "${AUDIT}/overlay_sparse_occ_centers_on_dense_20260824.py" \
      --sparse-pkl "${SPARSE_GT}" --sparse-root "${sparse}" --dense-root "${dense}" \
      >"${ROOT}/logs/overlay_occ${views}.log" 2>&1
  fi
done

run_variant() (
  set -euo pipefail
  local variant=$1 gpu=$2
  local images=${ROOT}/${variant}/images
  local run=${ROOT}/${variant}/frontends
  local hrnet=${run}/hrnet
  local resnet=${run}/resnet152
  local hrnet_type=posefusion_${variant}_dense_hrnet
  local resnet_type=posefusion_${variant}_dense_resnet152
  local hrnet_type_dir=${DATA}/data/datasets_mmpose/annot_temporal_5_5_${hrnet_type}
  local resnet_type_dir=${DATA}/data/datasets_mmpose/annot_temporal_5_5_${resnet_type}
  mkdir -p "${hrnet}/shards" "${hrnet}/logs" "${hrnet}/merged" \
    "${resnet}" "${hrnet_type_dir}" "${resnet_type_dir}"
  test -s "${ROOT}/${variant}/protocol_manifest.json"

  run_hrnet() {
    local pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      local output=${hrnet}/shards/shard${shard}.pkl
      local manifest=${output}.manifest.json
      if [[ -s "${output}" && -s "${manifest}" ]]; then continue; fi
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "${PY}" -u "${AUDIT}/export_h36m_gbt_aligned_hrnet_20260814.py" \
          --input-pkl "${GT}" --images-root "${images}" \
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
    (( failed == 0 ))
    local completed=("${hrnet}"/shards/shard*.pkl)
    [[ ${#completed[@]} -eq ${NUM_SHARDS} ]]
    "${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
      --input-pkl "${GT}" --shards "${completed[@]}" \
      --output "${hrnet}/merged/h36m_validation.pkl" \
      --manifest "${hrnet}/merged/h36m_validation.manifest.json" \
      >"${hrnet}/logs/merge.log" 2>&1
    ln -sfn "${hrnet}/merged/h36m_validation.pkl" "${hrnet_type_dir}/h36m_validation.pkl"
    date --iso-8601=seconds >"${hrnet}/COMPLETED"
  }

  run_resnet() {
    if [[ ! -s "${resnet}/h36m_validation.pkl" || ! -s "${resnet}/report.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "${PY}" -u "${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py" \
          --pkl "${GT}" --image-root "${images}" --checkpoint "${LT_CHECKPOINT}" \
          --config "${LT_CONFIG}" --output "${resnet}/report.json" \
          --export-rumpl-pkl "${resnet}/h36m_validation.pkl" \
          --batch-size 32 --workers 8 --device cuda:0 \
          >"${resnet}/export.log" 2>&1
    fi
    ln -sfn "${resnet}/h36m_validation.pkl" "${resnet_type_dir}/h36m_validation.pkl"
    date --iso-8601=seconds >"${resnet}/COMPLETED"
  }

  run_hrnet & p0=$!
  run_resnet & p1=$!
  failed=0
  wait "${p0}" || failed=1
  wait "${p1}" || failed=1
  (( failed == 0 ))
  sha256sum "${hrnet}/merged/h36m_validation.pkl" "${resnet}/h36m_validation.pkl" \
    >"${run}/sha256.txt"
  date --iso-8601=seconds >"${run}/COMPLETED"
  echo "[${variant} dense frontends] complete"
)

if [[ "${SERIAL_VARIANTS}" == 1 ]]; then
  run_variant occ2 "${OCC2_GPU}"
  run_variant occ3 "${OCC3_GPU}"
else
  run_variant occ2 "${OCC2_GPU}" & p2=$!
  run_variant occ3 "${OCC3_GPU}" & p3=$!
  failed=0
  wait "${p2}" || failed=1
  wait "${p3}" || failed=1
  (( failed == 0 ))
fi
date --iso-8601=seconds >"${ROOT}/frontends_COMPLETED"
echo "[dense VOC Occ-2/Occ-3 frontends] complete"
