#!/usr/bin/env bash
# Generate the dense S9/S11 ResNet-152 frontend required by T=9 temporal
# validation.  The ordinary H36M validation pkl is intentionally sparse
# (image_id step 65); this uses the original dense GT record (step 5), rather
# than an HRNet-exported record whose distortion fields were already cleared.
# The official LT preprocessing and checkpoint are otherwise unchanged.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
INPUT=${DATA_ROOT}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
IMAGES=${DATA_ROOT}/images
CHECKPOINT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/pretrained/pose_resnet_4.5_pixels_human36m_mmpose.pth
CONFIG=/home/lixiaob/cjy/reference/learnable-triangulation-official/experiments/human36m/eval/human36m_alg.yaml
OUT=/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/frontend_temporal_v2_gtinput/validation
PKL=${OUT}/h36m_validation_res152_temporal.pkl
REPORT=${OUT}/report.json

mkdir -p "${OUT}"
test -s "${INPUT}" && test -s "${IMAGES}" && test -s "${CHECKPOINT}" && test -s "${CONFIG}"
if [[ ! -s "${PKL}" || ! -s "${REPORT}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/eval_lt_official_on_rumpl_h36m_20260813.py" \
    --pkl "${INPUT}" --image-root "${IMAGES}" --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" --output "${REPORT}" --export-only \
    --export-rumpl-pkl "${PKL}" --batch-size 32 --workers 8 --device cuda:0 \
    >"${OUT}/export.log" 2>&1
fi
test -s "${PKL}" && test -s "${REPORT}"
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[ResNet-temporal-frontend] complete ${PKL}"
