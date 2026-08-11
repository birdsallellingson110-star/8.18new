#!/usr/bin/env bash
# H108-H111: retained H76 + parallel joint-query residual decoder.
# Unlike H17, the query decoder does not replace RUMPL VFT/PFT/head.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/H108_H111_parallel_joint_query

mkdir -p "${OUT}/logs"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

common=(
  --cfg "${CFG}" --base-checkpoint "${BASE}"
  --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64
  --window-length 1 --frame-stride 5 --num-views 2 --depth 3 --heads 8
  --fusion-mode query-residual --residual-scale 1.0
  --optimizer-steps 5000 --warmup-steps 500
  --micro-batch-size 4 --effective-batch-size 32 --workers 3
  --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0
  --amp-dtype bf16 --log-every 50 --save-every 1000
  --disable-missing-keypoints --loss-type mpjpe --loss-frame latest
)

train_one() {
  local name=$1
  shift
  mkdir -p "${OUT}/${name}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
    "${common[@]}" --output-dir "${OUT}/${name}" "$@" \
    >"${OUT}/logs/${name}.log" 2>&1
}

train_one H108_query_unbiased_drop0 --token-dropout 0.0 & p0=$!
train_one H109_query_biased_drop0 --token-dropout 0.0 --biased & p1=$!
train_one H110_query_unbiased_drop20 --token-dropout 0.2 & p2=$!
train_one H111_query_biased_drop20 --token-dropout 0.2 --biased & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"
date --iso-8601=seconds >"${OUT}/training.done"

eval_one() {
  local name=$1 biased=$2 views=$3
  local output=${OUT}/eval/${name}/V${views}
  mkdir -p "${output}"
  local extra=()
  [[ "${biased}" == 1 ]] && extra+=(--biased)
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${OUT}/${name}/checkpoint_step_0005000.pth" \
    "${extra[@]}" --fusion-mode query-residual --residual-scale 1.0 \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 1 --frame-stride 5 --depth 3 --heads 8 \
    --batch-size 64 --workers 4 --device cuda:0 >"${output}/eval.log" 2>&1
}

for views in 2 3 4; do
  eval_one H108_query_unbiased_drop0 0 "${views}" & q0=$!
  eval_one H109_query_biased_drop0 1 "${views}" & q1=$!
  eval_one H110_query_unbiased_drop20 0 "${views}" & q2=$!
  eval_one H111_query_biased_drop20 1 "${views}" & q3=$!
  wait "${q0}"
  wait "${q1}"
  wait "${q2}"
  wait "${q3}"
done
date --iso-8601=seconds >"${OUT}/completed.done"
