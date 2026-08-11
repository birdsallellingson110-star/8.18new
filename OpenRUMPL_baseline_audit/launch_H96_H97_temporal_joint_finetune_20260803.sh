#!/usr/bin/env bash
set -euo pipefail
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python; REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL; ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml; BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar; TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap; OUT=${ROOT}/H96_H97_temporal_joint_finetune; WAIT=${ROOT}/H93_H95_temporal_eval/completed.done
mkdir -p "${OUT}/logs" "${OUT}/H96_unbiased" "${OUT}/H97_biased"; exec 9>"${OUT}/pipeline.lock"; flock 9
while [[ ! -s "${WAIT}" ]]; do sleep 30; done
export PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
gpu_h96=1
if ! nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  gpu_h96=0
fi
echo "H96 physical GPU=${gpu_h96}; H97 physical GPU=1" >"${OUT}/gpu_assignment.log"
common=(--cfg "${CFG}" --base-checkpoint "${BASE}" --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64 --window-length 9 --frame-stride 5 --num-views 2 --depth 3 --heads 8 --token-dropout 0.2 --optimizer-steps 3000 --warmup-steps 300 --micro-batch-size 1 --effective-batch-size 8 --workers 4 --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 --amp-dtype bf16 --log-every 50 --save-every 1000 --unfreeze-backbone --backbone-lr-multiplier 0.05)
CUDA_VISIBLE_DEVICES="${gpu_h96}" "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" "${common[@]}" --output-dir "${OUT}/H96_unbiased" >"${OUT}/logs/H96.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" "${common[@]}" --biased --output-dir "${OUT}/H97_biased" >"${OUT}/logs/H97.log" 2>&1 & p1=$!
wait "${p0}"; wait "${p1}"; date --iso-8601=seconds >"${OUT}/completed.done"
