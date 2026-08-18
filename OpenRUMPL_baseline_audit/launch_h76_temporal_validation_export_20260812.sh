#!/usr/bin/env bash
set -euo pipefail

shard=${1:?usage: $0 <shard:0|1> <physical-gpu> [max-groups]}
gpu=${2:?usage: $0 <shard:0|1> <physical-gpu> [max-groups]}
max_groups=${3:-0}
if [[ "${shard}" != "0" && "${shard}" != "1" ]]; then
  echo "shard must be 0 or 1" >&2; exit 2
fi
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
OUTROOT=${ROOT}/Temporal_H76_Hypotheses_20260812
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT=$(tr -d '\r\n' <${ROOT}/H76_h50_centered_plucker/checkpoints/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803.txt)
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
mkdir -p "${OUTROOT}"
suffix=""
if [[ "${max_groups}" -gt 0 ]]; then suffix="_smoke${max_groups}"; fi
OUT=${OUTROOT}/H76_temporal_validation_all_subsets_shard${shard}of2${suffix}.npz
LOG=${OUTROOT}/H76_temporal_validation_all_subsets_shard${shard}of2${suffix}.log

export CUDA_VISIBLE_DEVICES="${gpu}"
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_SEMANTIC_GRAPH_PRE_VFT=off
export RUMPL_GRAFORMER_PFT=off
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib:${AUDIT}

"${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
  --cfg "${CFG}" --checkpoint "${CKPT}" \
  --dataset-name annot_temporal_5_5 --mmpose-type "${TYPE}" \
  --subset validation --output "${OUT}" --shard-index "${shard}" \
  --num-shards 2 --batch-size 128 --workers 12 \
  --max-groups "${max_groups}" --gpu "${gpu}" >"${LOG}" 2>&1

echo "saved ${OUT}"
