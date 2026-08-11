#!/usr/bin/env bash
# Root-protected MixSTE pose-space ablation.
# P1: causal T9, supervise latest frame only.
# P2: bidirectional T27, supervise all frames as in official MixSTE.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/MixSTE_pose_rootprotected_20260808

mkdir -p "${OUT}/logs"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "${CFG}" && test -s "${BASE}"

train_one() {
  local gpu=$1
  local name=$2
  local length=$3
  local loss_frame=$4
  mkdir -p "${OUT}/${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64 \
    --window-length "${length}" --frame-stride 5 --num-views 2 \
    --depth 4 --heads 8 --fusion-mode mixste-pose-residual \
    --residual-scale 1.0 --token-dropout 0.0 \
    --optimizer-steps 5000 --warmup-steps 500 \
    --micro-batch-size 1 --effective-batch-size 8 --workers 4 \
    --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 \
    --amp-dtype bf16 --log-every 50 --save-every 1000 \
    --disable-missing-keypoints --loss-profile rumpl --loss-type mpjpe \
    --loss-frame "${loss_frame}" --output-dir "${OUT}/${name}" \
    >"${OUT}/logs/${name}.log" 2>&1
}

train_one 0 P1_pose_root_T9_causal_latest 9 latest & p0=$!
train_one 1 P2_pose_root_T27_bidirectional_all 27 all & p1=$!
wait "${p0}"
wait "${p1}"
date --iso-8601=seconds >"${OUT}/training.done"
