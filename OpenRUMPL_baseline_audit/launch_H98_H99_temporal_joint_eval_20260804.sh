#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
TRAIN=${ROOT}/H96_H97_temporal_joint_finetune
OUT=${ROOT}/H98_H99_temporal_joint_eval

mkdir -p "${OUT}/logs"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

eval_one() {
  local gpu=$1 name=$2 checkpoint=$3 biased=$4 views=$5
  local output=${OUT}/${name}/V${views}
  mkdir -p "${output}"
  local extra=()
  [[ "${biased}" == 1 ]] && extra+=(--biased)
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${checkpoint}" "${extra[@]}" \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 3 --heads 8 \
    --batch-size 32 --workers 6 --device cuda:0 \
    >"${output}/eval.log" 2>&1
}

H96=${TRAIN}/H96_unbiased/checkpoint_step_0003000.pth
H97=${TRAIN}/H97_biased/checkpoint_step_0003000.pth
eval_one 0 H96_unbiased "${H96}" 0 2 & p0=$!
eval_one 0 H96_unbiased "${H96}" 0 4 & p1=$!
eval_one 1 H97_biased "${H97}" 1 2 & p2=$!
eval_one 1 H97_biased "${H97}" 1 4 & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"
date --iso-8601=seconds >"${OUT}/completed.done"
