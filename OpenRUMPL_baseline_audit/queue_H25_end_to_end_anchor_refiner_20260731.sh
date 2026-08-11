#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
VARIANT=${2:-joint}
WAIT_FOR=${3:-}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CACHE=${ROOT}/H24_h22_train_predictions
H21=${ROOT}/H21_pose_query_v2focus_reg005/final.pth
H24_GATE=${ROOT}/H24_learned_anchor_gate/final.pth
H22_EVAL=/mnt/data/cjyoutput/h36m_original_rumpl_tri_anchor_20260731/eval/H20_H22_CUR_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_clean_realH36M_seed0_20260731
OUTPUT=${ROOT}/H25_end_to_end_anchor_refiner_${VARIANT}

case "${VARIANT}" in
  joint)
    EXTRA_ARGS=()
    ;;
  frozen_gate)
    EXTRA_ARGS=(--freeze-gate)
    ;;
  *)
    echo "unknown H25 variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ -n "${WAIT_FOR}" ]]; then
  while [[ ! -s "${WAIT_FOR}" ]]; do
    sleep 15
  done
fi

mkdir -p "${OUTPUT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${AUDIT}"
if [[ ! -s "${OUTPUT}/h21_final.pth" \
      || ! -s "${OUTPUT}/gate_final.pth" ]]; then
  "${PY}" -u "${AUDIT}/train_h25_end_to_end_anchor_refiner.py" \
    --input-pkl \
      "${DATA}/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --rumpl-input-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_legswap/h36m_train.pkl" \
    --dense-shards "${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz \
    --h21-checkpoint "${H21}" \
    --gate-checkpoint "${H24_GATE}" \
    --rumpl-prediction-dicts \
      "${CACHE}/V2/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
      "${CACHE}/V3/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
      "${CACHE}/V4/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
    --group-manifests \
      "${CACHE}/V2/groups.json" \
      "${CACHE}/V3/groups.json" \
      "${CACHE}/V4/groups.json" \
    --steps 5000 \
    --view-probabilities 3 1 1 \
    --h21-learning-rate 0.00003 \
    --gate-learning-rate 0.0001 \
    --two-d-loss-weight 0.002 \
    --delta-penalty-weight 0.001 \
    --oracle-loss-weight 0.01 \
    --gate-drift-weight 0.01 \
    --seed 0 \
    --device cuda:0 \
    --output-dir "${OUTPUT}" \
    "${EXTRA_ARGS[@]}" \
    >"${OUTPUT}/train.log" 2>&1
fi

if [[ ! -s "${OUTPUT}/full_eval.json" ]]; then
  "${PY}" -u "${AUDIT}/eval_h23_rumpl_pose_query_anchor.py" \
    --input-pkl \
      "${DATA}/datasets/annot_filtered_5_64/h36m_validation.pkl" \
    --rumpl-input-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_inferencer_legswap/h36m_validation.pkl" \
    --dense-shards "${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz \
    --checkpoint "${OUTPUT}/h21_final.pth" \
    --gate-checkpoint "${OUTPUT}/gate_final.pth" \
    --prediction-root "${H22_EVAL}" \
    --views 2 3 4 \
    --query-sources old_anchor \
    --anchor-delta-scales 0.25 \
    --device cuda:0 \
    --output "${OUTPUT}/full_eval.json" \
    >"${OUTPUT}/full_eval.log" 2>&1
fi

echo "[H25] complete variant=${VARIANT} $(date --iso-8601=seconds)"
