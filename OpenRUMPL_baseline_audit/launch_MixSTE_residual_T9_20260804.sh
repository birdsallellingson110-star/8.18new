#!/usr/bin/env bash
# Stable residual MixSTE TTB follow-up.  Invoke only after the strict TTB run
# frees GPU0; GPU1 remains reserved for another user's process.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/MixSTE_residual_T9_20260804

mkdir -p "${OUT}/logs" "${OUT}/eval"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "${CFG}" && test -s "${BASE}"

common=(
  --cfg "${CFG}" --base-checkpoint "${BASE}"
  --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64
  --window-length 9 --frame-stride 5 --num-views 2 --depth 4 --heads 8
  --fusion-mode mixste-ttb-residual --residual-scale 0.1 --token-dropout 0.0
  --optimizer-steps 3000 --warmup-steps 300
  --micro-batch-size 1 --effective-batch-size 8 --workers 3
  --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0
  --amp-dtype bf16 --log-every 50 --save-every 1000
  --disable-missing-keypoints --loss-type mpjpe --loss-frame all
)

train_one() {
  local name=$1
  shift
  mkdir -p "${OUT}/${name}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
    "${common[@]}" --output-dir "${OUT}/${name}" "$@" \
    >"${OUT}/logs/${name}.log" 2>&1
}

train_one R1_ttb_residual_rumpl_mpjpe & p0=$!
train_one R2_ttb_residual_mixste_original --loss-profile mixste-original & p1=$!
wait "${p0}"
wait "${p1}"
date --iso-8601=seconds >"${OUT}/training.done"

eval_one() {
  local name=$1 views=$2
  local output=${OUT}/eval/${name}/V${views}
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${OUT}/${name}/checkpoint_step_0003000.pth" \
    --fusion-mode mixste-ttb-residual --residual-scale 0.1 \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 32 --workers 4 --device cuda:0 \
    >"${output}/eval.log" 2>&1
}

for views in 2 3 4; do
  eval_one R1_ttb_residual_rumpl_mpjpe "${views}" & q0=$!
  eval_one R2_ttb_residual_mixste_original "${views}" & q1=$!
  wait "${q0}"
  wait "${q1}"
done
date --iso-8601=seconds >"${OUT}/completed.done"
