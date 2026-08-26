#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: launch_h19_causal_seq2seq_variant_20260818.sh VARIANT PHYSICAL_GPU}
PHYSICAL_GPU=${2:?usage: launch_h19_causal_seq2seq_variant_20260818.sh VARIANT PHYSICAL_GPU}
case "${PHYSICAL_GPU}" in
  0|1) ;;
  *) echo "physical GPU must be 0 or 1" >&2; exit 2 ;;
esac

case "${VARIANT}" in
  h19a_protected)
    ROOT_MODE=protected
    TEMPORAL_WEIGHT=0.0
    OUT_NAME=h19a_causal_seq2seq_root_protected
    ;;
  h19b_learned_root)
    ROOT_MODE=learned
    TEMPORAL_WEIGHT=0.0
    OUT_NAME=h19b_causal_seq2seq_root_learned
    ;;
  h19c_learned_root_tloss)
    ROOT_MODE=learned
    TEMPORAL_WEIGHT=0.10
    OUT_NAME=h19c_causal_seq2seq_root_learned_tloss
    ;;
  *)
    echo "unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818
TRAIN_CACHE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/train_c2_22c.npz
TRAIN_FUSED=${ROOT}/h18_clean_temporal_cache/train/fused_poses.npy
TRAIN_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_CACHE=${ROOT}/h15_temporal_c2_oracle/validation_c2_22c.npz
VAL_FUSED=${ROOT}/h18_clean_temporal_cache/validation/fused_poses.npy
VAL_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl
OUT=${ROOT}/${OUT_NAME}
if [[ ! -s "${VAL_PKL}" ]]; then
  echo "missing validation pkl: ${VAL_PKL}" >&2
  exit 2
fi
if [[ ! -s "${TRAIN_PKL}" || ! -s "${TRAIN_FUSED}" || ! -s "${VAL_FUSED}" ]]; then
  echo "missing H19 cache or train pkl" >&2
  exit 2
fi

export PYTHONPATH=${AUDIT}
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "${OUT}"

if [[ ! -s "${OUT}/COMPLETED" ]]; then
  "${PY}" -u "${AUDIT}/train_e2_causal_temporal_seq2seq_20260818.py" \
    --train-cache "${TRAIN_CACHE}" --train-fused "${TRAIN_FUSED}" \
    --train-pkl "${TRAIN_PKL}" \
    --validation-cache "${VAL_CACHE}" --validation-fused "${VAL_FUSED}" \
    --validation-pkl "${VAL_PKL}" --output-dir "${OUT}" \
    --window-length 9 --frame-stride 5 --epochs 12 --batch-size 64 \
    --hidden-dim 96 --layers 2 --lr 5e-5 --weight-decay 5e-4 \
    --relative-scale-m 0.10 --root-scale-m 0.05 \
    --root-mode "${ROOT_MODE}" --temporal-loss-weight "${TEMPORAL_WEIGHT}" \
    --gpu 0 --seed 0 >"${OUT}/train.log" 2>&1
fi

date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "${VARIANT} complete: ${OUT}"
