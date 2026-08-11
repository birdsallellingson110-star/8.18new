#!/usr/bin/env bash
# Joint adaptation screening after the frozen-backbone MixSTE ablation.
# Keep RUMPL ray/VFT geometry fixed, but allow its PFT/head to co-adapt with
# the MixSTE TTB.  This is the minimal relaxation needed to compare against
# MixSTE's official end-to-end STB/TTB training without replacing RUMPL.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/MixSTE_joint_pft_T9_20260804

mkdir -p "${OUT}/logs"
exec 9>"${OUT}/pipeline.lock"
flock 9
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -s "${CFG}" && test -s "${BASE}"

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
  --unfreeze-backbone --backbone-train-scope pft-head
  --backbone-lr-multiplier 0.1 --backbone-eval-mode
)

train_one() {
  local name=$1
  shift
  mkdir -p "${OUT}/${name}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
    "${common[@]}" --output-dir "${OUT}/${name}" "$@" \
    >"${OUT}/logs/${name}.log" 2>&1
}

train_one J1_ttb_pft_rumpl_mpjpe & p0=$!
train_one J2_ttb_pft_mixste_original --loss-profile mixste-original & p1=$!
wait "${p0}"
wait "${p1}"
date --iso-8601=seconds >"${OUT}/training.done"
