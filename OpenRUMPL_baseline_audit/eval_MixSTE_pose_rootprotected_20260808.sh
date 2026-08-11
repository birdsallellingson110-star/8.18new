#!/usr/bin/env bash
# Matched evaluation for root-protected pose-space MixSTE.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
RUN=${ROOT}/MixSTE_pose_rootprotected_20260808
CACHE_ROOT=${ROOT}/MixSTE_T9_strict_20260804/eval_fast
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap

mkdir -p "${RUN}/eval_fast"
exec 9>"${RUN}/eval_fast/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "${CFG}" && test -s "${BASE}"

eval_one() {
  local gpu=$1
  local name=$2
  local views=$3
  local length=$4
  local output_frame=$5
  local checkpoint=$6
  local output=${RUN}/eval_fast/${name}/V${views}
  local cache=${CACHE_ROOT}/cache_V${views}.pt
  mkdir -p "${output}"
  test -s "${checkpoint}" && test -s "${cache}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${checkpoint}" \
    --fusion-mode mixste-pose-residual --output-dir "${output}" \
    --mmpose-type "${TYPE}" --dataset-name annot_temporal_5_5 \
    --num-views "${views}" --window-length "${length}" --frame-stride 5 \
    --output-frame "${output_frame}" --depth 4 --heads 8 \
    --residual-scale 1.0 --batch-size 128 --workers 8 --device cuda:0 \
    --frame-cache "${cache}" --cache-workers 0 \
    >"${output}/eval.log" 2>&1
}

eval_baseline_center() {
  local gpu=$1
  local views=$2
  local output=${RUN}/eval_fast/M0_H76_T27_center/V${views}
  local cache=${CACHE_ROOT}/cache_V${views}.pt
  mkdir -p "${output}"
  test -s "${cache}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" --backbone-only \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 27 --frame-stride 5 --output-frame center \
    --batch-size 128 --workers 8 --device cuda:0 \
    --frame-cache "${cache}" --cache-workers 0 \
    >"${output}/eval.log" 2>&1
}

P1=${RUN}/P1_pose_root_T9_causal_latest/checkpoint_step_0005000.pth
P2=${RUN}/P2_pose_root_T27_bidirectional_all/checkpoint_step_0005000.pth

# One view-cardinality stream per GPU at a time; run V4 first for an early read.
for views in 4 3 2; do
  eval_one 0 P1_pose_root_T9_causal_latest "${views}" 9 latest "${P1}" & p0=$!
  eval_one 1 P2_pose_root_T27_bidirectional_all "${views}" 27 center "${P2}" & p1=$!
  wait "${p0}"
  wait "${p1}"
  eval_baseline_center 1 "${views}"
done
date --iso-8601=seconds >"${RUN}/eval_fast.done"
