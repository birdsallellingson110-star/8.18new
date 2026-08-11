#!/usr/bin/env bash
# Evaluate J1/J2 with the same cached, causal Table-2 protocol.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
OUT=${ROOT}/MixSTE_joint_pft_T9_20260804
EVAL_ROOT=${ROOT}/MixSTE_T9_strict_20260804/eval_fast
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap

mkdir -p "${OUT}/eval_fast/logs"
exec 9>"${OUT}/eval_fast/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "${CFG}" && test -s "${BASE}"

eval_one() {
  local name=$1
  local views=$2
  local checkpoint=$3
  local output=${OUT}/eval_fast/${name}/V${views}
  local cache=${EVAL_ROOT}/cache_V${views}.pt
  mkdir -p "${output}"
  test -s "${checkpoint}" && test -s "${cache}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${checkpoint}" \
    --fusion-mode mixste-ttb \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 128 --workers 12 --device cuda:0 \
    --frame-cache "${cache}" --cache-workers 0 \
    >"${output}/eval.log" 2>&1
}

J1=${OUT}/J1_ttb_pft_rumpl_mpjpe/checkpoint_step_0005000.pth
J2=${OUT}/J2_ttb_pft_mixste_original/checkpoint_step_0005000.pth
declare -a p1 p2
for views in 2 3 4; do
  eval_one J1_ttb_pft_rumpl_mpjpe "${views}" "${J1}" & p1[${views}]=$!
  eval_one J2_ttb_pft_mixste_original "${views}" "${J2}" & p2[${views}]=$!
done
for views in 2 3 4; do
  wait "${p1[${views}]}"
  wait "${p2[${views}]}"
done
date --iso-8601=seconds >"${OUT}/eval_fast.done"
