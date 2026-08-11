#!/usr/bin/env bash
# MixSTE adaptation screening on the real-H36M RUMPL baseline.
#
# The retained RUMPL ray encoder, VFT, PFT and 3-D head are unchanged.  The
# only architectural insertion is a MixSTE-style per-joint TTB after VFT and
# before the original PFT.  M1/M2 separate architecture from objective:
# M1 uses ordinary all-frame RUMPL MPJPE; M2 uses the official MixSTE H36M
# weighted-MPJPE + temporal-consistency loss.  GPU1 is intentionally unused:
# it belongs to another user's RayMixSTE process.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/MixSTE_T9_strict_20260804

mkdir -p "${OUT}/logs" "${OUT}/eval"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

test -s "${CFG}"
test -s "${BASE}"

common=(
  --cfg "${CFG}" --base-checkpoint "${BASE}"
  --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64
  --window-length 9 --frame-stride 5 --num-views 2 --depth 4 --heads 8
  --fusion-mode mixste-ttb --token-dropout 0.0
  --optimizer-steps 5000 --warmup-steps 500
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

train_one M1_mixste_ttb_rumpl_mpjpe & p0=$!
train_one M2_mixste_ttb_mixste_original --loss-profile mixste-original & p1=$!
wait "${p0}"
wait "${p1}"
date --iso-8601=seconds >"${OUT}/training.done"

eval_one() {
  local name=$1 views=$2
  local output=${OUT}/eval/${name}/V${views}
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${OUT}/${name}/checkpoint_step_0005000.pth" \
    --fusion-mode mixste-ttb \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 32 --workers 4 --device cuda:0 \
    >"${output}/eval.log" 2>&1
}

# The no-checkpoint run is the exact H76 gate/zero-projection control under
# the same T=9 evaluator.  It must be recorded before interpreting M1/M2.
for views in 2 3 4; do
  output=${OUT}/eval/M0_H76_identity/V${views}
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --fusion-mode mixste-ttb \
    --output-dir "${output}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 32 --workers 4 --device cuda:0 \
    >"${output}/eval.log" 2>&1
done

for views in 2 3 4; do
  eval_one M1_mixste_ttb_rumpl_mpjpe "${views}" & q0=$!
  eval_one M2_mixste_ttb_mixste_original "${views}" & q1=$!
  wait "${q0}"
  wait "${q1}"
done
date --iso-8601=seconds >"${OUT}/completed.done"
