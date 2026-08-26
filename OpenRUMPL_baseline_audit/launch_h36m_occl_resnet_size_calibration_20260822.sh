#!/usr/bin/env bash
# Reconstruct the only missing GBT H36M-Occl parameter (white-square size)
# against GBT's published Algebraic Triangulation control.  The three arms use
# identical deterministic masks; only square side changes.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
PKL=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
IMAGES=${DATA}/images
CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
CONFIG=/home/lixiaob/cjy/reference/learnable-triangulation-official/experiments/human36m/eval/human36m_alg.yaml
ROOT=/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/resnet_size_calibration
SEED=20260822

mkdir -p "${ROOT}"
test -s "${PKL}" && test -d "${IMAGES}"
test -s "${CHECKPOINT}" && test -s "${CONFIG}"
test -s "${AUDIT}/h36m_occlusion_protocol_20260822.py"
test -s "${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py"

run_arm() {
  local gpu=$1
  local tag=$2
  local fraction=$3
  local out=${ROOT}/${tag}
  mkdir -p "${out}"
  if [[ -s "${out}/result.json" && -s "${out}/h36m_validation_res152_occl.pkl" ]]; then
    echo "[H36M-Occl calibration] ${tag} already complete"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "${PY}" -u "${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py" \
      --pkl "${PKL}" --image-root "${IMAGES}" \
      --checkpoint "${CHECKPOINT}" --config "${CONFIG}" \
      --output "${out}/result.json" \
      --export-rumpl-pkl "${out}/h36m_validation_res152_occl.pkl" \
      --batch-size 32 --workers 6 --device cuda:0 \
      --occlusion-prob 0.1 --occlusion-square-fraction "${fraction}" \
      --occlusion-seed "${SEED}" \
      >"${out}/run.log" 2>&1
  sha256sum "${out}/result.json" "${out}/h36m_validation_res152_occl.pkl" \
    >"${out}/sha256.txt"
  date --iso-8601=seconds >"${out}/COMPLETED"
}

# Two GPUs remain occupied; GPU0 performs the third arm after f010 finishes.
(run_arm 0 f010 0.10; run_arm 0 f020 0.20) &
pid0=$!
(run_arm 1 f015 0.15) &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[H36M-Occl calibration] all arms complete"
