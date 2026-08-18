#!/usr/bin/env bash
# Build the dense T=9 validation cache with exactly the current GBT-aligned
# YOLOX-X + HRNet-W32 coordinate frontend.  This is intentionally separate
# from the older A1D/H21 temporal cache.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
AUDIT_SCRIPT=${AUDIT}/audit_h8_temporal_validation_20260817.py
COMBINE_SCRIPT=${AUDIT}/combine_h8_shard0_parts_20260817.py
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
IMAGES=${DATA}/images
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend
RUN=${OUT}/validation
SHARDS=${RUN}/shards
MERGED=${RUN}/merged/h36m_validation.pkl
MANIFEST=${RUN}/merged/h36m_validation.manifest.json
TYPE=gbt_yolox_x_score001_fallback_legswap
TYPE_DIR=${DATA}/data/datasets_mmpose/annot_temporal_5_5_${TYPE}

POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
DET_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_x_8xb8-300e_coco.py
DET_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
NUM_SHARDS=${NUM_SHARDS:-4}

mkdir -p "${SHARDS}" "${RUN}/logs" "${RUN}/merged" "${TYPE_DIR}"
test -s "${GT}" || { echo "missing dense GT: ${GT}" >&2; exit 1; }
test -s "${AUDIT_SCRIPT}"
test -s "${COMBINE_SCRIPT}"
test -s "${DET_CONFIG}" && test -s "${DET_CHECKPOINT}"
test -s "${POSE_CONFIG}" && test -s "${POSE_CHECKPOINT}"

# Recover an interrupted strict shard from two explicit half-shards.  The
# exporter writes pickle files atomically, so this branch is safe while the
# half-shards are still running (it simply waits for both files to exist).
if [[ ! -s "${SHARDS}/shard0.pkl" \
      && -s "${SHARDS}/part0.pkl" && -s "${SHARDS}/part4.pkl" \
      && -s "${SHARDS}/part0.manifest.json" \
      && -s "${SHARDS}/part4.manifest.json" ]]; then
  "${PY}" -u "${COMBINE_SCRIPT}" \
    --parts "${SHARDS}/part0.pkl" "${SHARDS}/part4.pkl" \
    --manifests "${SHARDS}/part0.manifest.json" "${SHARDS}/part4.manifest.json" \
    --output "${SHARDS}/shard0.pkl" \
    --manifest "${SHARDS}/shard0.manifest.json" \
    >"${RUN}/logs/combine_shard0.log" 2>&1
fi

if [[ -s "${MERGED}" && -s "${MANIFEST}" ]]; then
  echo "[H8-EXPORT] merged cache already exists: ${MERGED}"
else
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=$((shard % 2))
    output=${SHARDS}/shard${shard}.pkl
    manifest=${SHARDS}/shard${shard}.manifest.json
    if [[ -s "${output}" && -s "${manifest}" ]]; then
      echo "[H8-EXPORT] shard ${shard} already complete"
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
        >"${RUN}/logs/shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if (( failed )); then
    echo "[H8-EXPORT] one or more shards failed; inspect ${RUN}/logs" >&2
    exit 1
  fi
  shards=("${SHARDS}"/shard*.pkl)
  if [[ ${#shards[@]} -ne ${NUM_SHARDS} ]]; then
    echo "[H8-EXPORT] expected ${NUM_SHARDS} shards, got ${#shards[@]}" >&2
    exit 1
  fi
  "${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
    --input-pkl "${GT}" --shards "${shards[@]}" \
    --output "${MERGED}" --manifest "${MANIFEST}" \
    >"${RUN}/logs/merge.log" 2>&1
fi

"${PY}" -u "${AUDIT_SCRIPT}" \
  --input-pkl "${MERGED}" --manifest "${MANIFEST}" \
  --output-json "${RUN}/audit.json" \
  --max-fallbacks 64 \
  >"${RUN}/logs/audit.log" 2>&1

target=${TYPE_DIR}/h36m_validation.pkl
if [[ -e "${target}" || -L "${target}" ]]; then
  existing=$(readlink -f "${target}" || true)
  expected=$(readlink -f "${MERGED}")
  [[ "${existing}" == "${expected}" ]] || {
    echo "[H8-EXPORT] refusing to replace ${target} -> ${existing}" >&2
    exit 1
  }
else
  ln -s "${MERGED}" "${target}"
fi

printf '%s\n' "${MERGED}" >"${RUN}/path.txt"
date --iso-8601=seconds >"${OUT}/validation_complete.done"
echo "[H8-EXPORT] complete: ${MERGED}"
