#!/usr/bin/env bash
# Launch one Stage-D variant on one physical GPU.
set -euo pipefail

variant=${1:?usage: $0 <geometry|utility> <physical-gpu> [smoke-batches]}
gpu=${2:?usage: $0 <geometry|utility> <physical-gpu> [smoke-batches]}
smoke=${3:-0}
if [[ "${variant}" != "geometry" && "${variant}" != "utility" ]]; then
  echo "variant must be geometry or utility" >&2
  exit 2
fi

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
STAGE=${ROOT}/Counterfactual_View_Utility_20260811
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT=$(tr -d '\r\n' <${ROOT}/H76_h50_centered_plucker/checkpoints/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803.txt)
C4=${STAGE}/C4_ght_finetune/ght/model_best.pth.tar
TRAIN0=${STAGE}/train_hypotheses/H76_train_all_subsets_shard0of2.npz
TRAIN1=${STAGE}/train_hypotheses/H76_train_all_subsets_shard1of2.npz
VAL=${STAGE}/H76_validation_all_subsets.npz
suffix=""
if [[ "${smoke}" -gt 0 ]]; then suffix="_smoke${smoke}"; fi
OUT=${STAGE}/D_${variant}_ray_correction${suffix}
mkdir -p "${OUT}"

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

"${PY}" -u "${AUDIT}/train_selective_ray_corrector_20260811.py" \
  --cfg "${CFG}" --h76-checkpoint "${CKPT}" --c4-checkpoint "${C4}" \
  --train-shards "${TRAIN0}" "${TRAIN1}" --validation-cache "${VAL}" \
  --variant "${variant}" --output-dir "${OUT}" --epochs 5 \
  --batch-size 512 --workers 4 --holdout-modulo 40 --seed 0 \
  --max-angle-degrees 0.5 --angle-regularizer 1e-4 \
  --smoke-batches "${smoke}" >"${OUT}/train.log" 2>&1

echo "finished ${variant}: ${OUT}"
