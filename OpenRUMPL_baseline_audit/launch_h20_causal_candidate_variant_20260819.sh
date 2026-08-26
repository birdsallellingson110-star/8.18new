#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: VARIANT PHYSICAL_GPU}
PHYSICAL_GPU=${2:?usage: VARIANT PHYSICAL_GPU}
case "${PHYSICAL_GPU}" in 0|1) ;; *) echo "gpu must be 0 or 1" >&2; exit 2 ;; esac

GEOM=""
V2_WEIGHT=1.0
TLOSS=0.0
ENCODER=mixste
case "${VARIANT}" in
  h20a_mixste)
    OUT_NAME=h20a_causal_candidate_mixste
    ;;
  h20b_geom_gate)
    OUT_NAME=h20b_causal_candidate_geom_gate
    GEOM=--geometry-gate
    V2_WEIGHT=2.0
    ;;
  h20c_tloss)
    OUT_NAME=h20c_causal_candidate_tloss
    TLOSS=0.10
    ;;
  h20d_joint)
    OUT_NAME=h20d_causal_candidate_joint
    ENCODER=joint
    ;;
  *)
    echo "unknown variant ${VARIANT}" >&2
    exit 2
    ;;
esac

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818
H16=${ROOT}/h16_temporal_c2_screen
TRAIN_CACHE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/train_c2_22c.npz
VAL_CACHE=${ROOT}/h15_temporal_c2_oracle/validation_c2_22c.npz
TRAIN_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL_PKL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl
OUT=${ROOT}/${OUT_NAME}

export PYTHONPATH=${AUDIT}
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "${OUT}"
test -s "${TRAIN_CACHE}" && test -s "${VAL_CACHE}" && test -s "${TRAIN_PKL}" && test -s "${VAL_PKL}"
test -s "${H16}/train_e2_scores.npy" && test -s "${H16}/validation_e2_scores.npy"

if [[ ! -s "${OUT}/COMPLETED" ]]; then
  "${PY}" -u "${AUDIT}/train_e2_causal_candidate_temporal_20260819.py" \
    --train-cache "${TRAIN_CACHE}" \
    --train-pkl "${TRAIN_PKL}" --train-scores "${H16}/train_e2_scores.npy" \
    --validation-cache "${VAL_CACHE}" --validation-pkl "${VAL_PKL}" \
    --validation-scores "${H16}/validation_e2_scores.npy" \
    --output-dir "${OUT}" --window-length 9 --frame-stride 5 \
    --train-window-stride 3 --encoder "${ENCODER}" \
    --epochs 8 --batch-size 32 --hidden-dim 64 --layers 2 --dropout 0.10 \
    --lr 5e-5 --weight-decay 5e-4 --identity-weight 0.5 \
    --v2-loss-weight "${V2_WEIGHT}" --temporal-loss-weight "${TLOSS}" \
    ${GEOM} --gpu 0 --seed 0 >"${OUT}/train.log" 2>&1
fi
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "${VARIANT} complete: ${OUT}"
