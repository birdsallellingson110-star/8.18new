#!/usr/bin/env bash
# Export two disjoint H36M-train hypothesis shards with the frozen H76 model.
set -euo pipefail

shard=${1:?usage: $0 <shard:0|1> <physical-gpu> [max-groups]}
gpu=${2:?usage: $0 <shard:0|1> <physical-gpu> [max-groups]}
max_groups=${3:-0}
if [[ "${shard}" != "0" && "${shard}" != "1" ]]; then
  echo "shard must be 0 or 1" >&2
  exit 2
fi

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
OUT=${ROOT}/Counterfactual_View_Utility_20260811/train_hypotheses
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT_FILE=${ROOT}/H76_h50_centered_plucker/checkpoints/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803.txt
CKPT=$(tr -d '\r\n' <"${CKPT_FILE}")
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
mkdir -p "${OUT}"

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

suffix=""
if [[ "${max_groups}" -gt 0 ]]; then suffix="_smoke${max_groups}"; fi
output=${OUT}/H76_train_all_subsets_shard${shard}of2${suffix}.npz
log=${OUT}/H76_train_all_subsets_shard${shard}of2${suffix}.log

"${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
  --cfg "${CFG}" --checkpoint "${CKPT}" \
  --dataset-name annot_filtered_5_64 --mmpose-type "${TYPE}" \
  --output "${output}" --shard-index "${shard}" --num-shards 2 \
  --batch-size 128 --workers 12 --max-groups "${max_groups}" --gpu "${gpu}" \
  >"${log}" 2>&1

echo "saved ${output}"
