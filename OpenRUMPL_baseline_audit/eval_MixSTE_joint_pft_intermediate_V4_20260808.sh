#!/usr/bin/env bash
# Screen all saved J1/J2 checkpoints on V4 before concluding that joint
# PFT/head adaptation is uniformly harmful. Uses the same cached Table-II
# evaluator as the final-checkpoint comparison.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
RUN=${ROOT}/MixSTE_joint_pft_T9_20260804
OUT=${RUN}/eval_fast_intermediate_V4
CACHE=${ROOT}/MixSTE_T9_strict_20260804/eval_fast/cache_V4.pt
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap

mkdir -p "${OUT}/logs"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "${CFG}" && test -s "${BASE}" && test -s "${CACHE}"

eval_one() {
  local gpu=$1
  local arm=$2
  local step=$3
  local checkpoint=${RUN}/${arm}/checkpoint_step_${step}.pth
  local output=${OUT}/${arm}/step_${step}
  mkdir -p "${output}"
  test -s "${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${checkpoint}" --fusion-mode mixste-ttb \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views 4 \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 128 --workers 8 --device cuda:0 \
    --frame-cache "${CACHE}" --cache-workers 0 \
    >"${output}/eval.log" 2>&1
}

# Run one checkpoint stream per GPU. Final step 5000 is already evaluated and
# is omitted; sequential checkpoints avoid CPU/disk oversubscription.
eval_arm() {
  local gpu=$1
  local arm=$2
  for step in 0001000 0002000 0003000 0004000; do
    eval_one "${gpu}" "${arm}" "${step}"
  done
}
eval_arm 0 J1_ttb_pft_rumpl_mpjpe & p0=$!
eval_arm 1 J2_ttb_pft_mixste_original & p1=$!
wait "${p0}"
wait "${p1}"
date --iso-8601=seconds >"${OUT}/eval.done"
