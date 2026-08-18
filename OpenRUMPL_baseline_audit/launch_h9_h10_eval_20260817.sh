#!/usr/bin/env bash
# Full clean H36M T=9 evaluation for H9/H10 after their checkpoints appear.
set -euo pipefail
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
VAL_NAME=annot_temporal_5_5
TYPE=gbt_yolox_x_score001_fallback_legswap
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817
FRAME_ROOT=${ROOT}/h8_pre_vft_temporal/frame_cache/H8_FROZEN

run_arm() {
  local arm=$1 gpu=$2 mode=$3
  local root=${ROOT}/${arm}
  local ckpt=${root}/checkpoint_step_0012000.pth
  while [[ ! -s "${ckpt}" ]]; do sleep 30; done
  for v in 2 3 4; do
    local dest=${root}/eval/T9/V${v}
    mkdir -p "${dest}"
    if [[ ! -s "${dest}/table2.json" ]]; then
      CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=${REPO}/lib \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
        --cfg "${CFG}" --base-checkpoint "${BASE}" --temporal-checkpoint "${ckpt}" \
        --backbone-flavor h76 --output-dir "${dest}" --mmpose-type "${TYPE}" \
        --dataset-name "${VAL_NAME}" --num-views "${v}" --window-length 9 \
        --model-window-length 9 --frame-stride 5 --output-frame latest \
        --fusion-mode "${mode}" --depth 4 --heads 8 --residual-scale 0.1 \
        --flip-lower-body-kp-test false --batch-size 16 --workers 6 \
        --frame-cache "${FRAME_ROOT}/V${v}.pt" --cache-workers 16 --device cuda:0 \
        >"${dest}/eval.log" 2>&1
      pred=$(find "${dest}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${pred}"
      "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
        --output-json "${dest}/table2.json" >"${dest}/table2.log" 2>&1
    fi
  done
  date --iso-8601=seconds >"${root}/eval_complete.done"
}

mkdir -p "${ROOT}/h9_mixste_pose_residual" "${ROOT}/h10_mixste_ttb_residual"
run_arm h9_mixste_pose_residual 0 mixste-pose-residual & p0=$!
run_arm h10_mixste_ttb_residual 1 mixste-ttb-residual & p1=$!
wait "${p0}"; wait "${p1}"
echo "H9/H10 eval complete $(date --iso-8601=seconds)"
