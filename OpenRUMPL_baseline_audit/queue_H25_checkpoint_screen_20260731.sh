#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
WAIT_FOR=${2:-}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
SOURCE=${ROOT}/H25_end_to_end_anchor_refiner_joint
OUTPUT=${ROOT}/H25_joint_checkpoint_screen
H22_EVAL=/mnt/data/cjyoutput/h36m_original_rumpl_tri_anchor_20260731/eval/H20_H22_CUR_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_clean_realH36M_seed0_20260731

if [[ -n "${WAIT_FOR}" ]]; then
  while [[ ! -s "${WAIT_FOR}" ]]; do
    sleep 15
  done
fi

mkdir -p "${OUTPUT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${AUDIT}"
for step in 1000 2000 3000 4000; do
  tag=$(printf "%06d" "${step}")
  result=${OUTPUT}/step${tag}_limit50.json
  if [[ -s "${result}" ]]; then
    continue
  fi
  "${PY}" -u "${AUDIT}/eval_h23_rumpl_pose_query_anchor.py" \
    --input-pkl \
      "${DATA}/datasets/annot_filtered_5_64/h36m_validation.pkl" \
    --rumpl-input-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_inferencer_legswap/h36m_validation.pkl" \
    --dense-shards "${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz \
    --checkpoint "${SOURCE}/h21_step${tag}.pth" \
    --gate-checkpoint "${SOURCE}/gate_step${tag}.pth" \
    --prediction-root "${H22_EVAL}" \
    --views 2 3 4 \
    --query-sources old_anchor \
    --anchor-delta-scales 0.25 \
    --limit-per-action 50 \
    --device cuda:0 \
    --output "${result}" \
    >"${OUTPUT}/step${tag}_limit50.log" 2>&1
done

echo "[H25 screen] complete $(date --iso-8601=seconds)"
