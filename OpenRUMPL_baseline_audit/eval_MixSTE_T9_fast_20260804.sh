#!/usr/bin/env bash
# Fast, protocol-identical evaluation for the completed MixSTE T=9 screening.
# Only the loader batch/workers differ from the strict evaluator; model,
# checkpoint, sampled windows, camera generator and Table-2 aggregation stay
# unchanged.  GPU1 is reserved for another user's process.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
OUT=${ROOT}/MixSTE_T9_strict_20260804
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap

mkdir -p "${OUT}/eval_fast/logs"
exec 9>"${OUT}/eval_fast/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

test -s "${CFG}"
test -s "${BASE}"

eval_one() {
  local name=$1
  local views=$2
  local checkpoint=$3
  local output=${OUT}/eval_fast/${name}/V${views}
  local cache=${OUT}/eval_fast/cache_V${views}.pt
  mkdir -p "${output}"
  local extra=()
  if [[ -n "${checkpoint}" ]]; then
    extra+=(--temporal-checkpoint "${checkpoint}")
  fi
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" "${extra[@]}" \
    --fusion-mode mixste-ttb \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 128 --workers 12 --device cuda:0 \
    --frame-cache "${cache}" --cache-workers 12 \
    >"${output}/eval.log" 2>&1
}

M1=${OUT}/M1_mixste_ttb_rumpl_mpjpe/checkpoint_step_0005000.pth
M2=${OUT}/M2_mixste_ttb_mixste_original/checkpoint_step_0005000.pth
test -s "${M1}"
test -s "${M2}"

declare -a p p1 p2
for views in 2 3 4; do
  eval_one M0_H76_identity "${views}" "" & p[${views}]=$!
done
for views in 2 3 4; do
  wait "${p[${views}]}"
done

for views in 2 3 4; do
  eval_one M1_mixste_ttb_rumpl_mpjpe "${views}" "${M1}" & p1[${views}]=$!
  eval_one M2_mixste_ttb_mixste_original "${views}" "${M2}" & p2[${views}]=$!
  wait "${p1[${views}]}"
  wait "${p2[${views}]}"
done
date --iso-8601=seconds >"${OUT}/eval_fast.done"
