#!/usr/bin/env bash
# After H173/H174 reach step 20k, run Table-II-aligned temporal eval (V2/V3/V4).
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
H81=$(tr -d '\r\n' < "${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt")
TRAIN=${ROOT}/H169_H174_temporal_h81_gbt_aligned
EVAL=${TRAIN}/eval
LOG=${TRAIN}/logs/queue_H173_H174_eval_20k.log

mkdir -p "${TRAIN}/logs"
exec >>"${LOG}" 2>&1
echo "[queue-eval] start $(date --iso-8601=seconds)"

wait_ckpt() {
  local path=$1
  while [[ ! -s "${path}" ]]; do
    echo "[queue-eval] waiting ${path} $(date --iso-8601=seconds)"
    sleep 120
  done
}

eval_run() {
  local gpu=$1 name=$2 ckpt=$3 fusion=$4 depth=$5 heads=$6 rscale=$7
  local extra=()
  shift 7
  extra=("$@")
  wait_ckpt "${ckpt}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export PYTHONPATH="${REPO}/lib"
  export RUMPL_EVAL_STRICT=0
  unset RUMPL_N_VIEWS_TRAIN_TEST_ALL
  local args=(--fusion-mode "${fusion}" --depth "${depth}" --heads "${heads}" --residual-scale "${rscale}" --biased "${extra[@]}")
  echo "[queue-eval] ${name} ckpt ready on GPU${gpu} $(date --iso-8601=seconds)"
  for v in 2 3 4; do
    local dest="${EVAL}/${name}/V${v}"
    mkdir -p "${dest}"
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
      --cfg "${CFG}" --base-checkpoint "${H81}" --backbone-flavor h81 \
      --temporal-checkpoint "${ckpt}" "${args[@]}" \
      --output-dir "${dest}" --mmpose-type "${TYPE}" \
      --dataset-name annot_temporal_5_5 --num-views "${v}" \
      --window-length 9 --frame-stride 5 \
      --batch-size 16 --workers 6 --device cuda:0 \
      >"${dest}/eval.log" 2>&1
    "${PY}" -c "import json; d=json.load(open('${dest}/table2.json')); print('[queue-eval]', '${name}', 'V${v}', d['windows'], round(d['table2_action_equal']['all17_mm'],2))"
  done
}

# H174 on GPU 0 (same card as its training; plenty of free VRAM).
eval_run 0 H174_mixsteTTB_mixsteLoss_trainV4 \
  "${TRAIN}/H174_mixsteTTB_mixsteLoss_trainV4/checkpoint_step_0020000.pth" \
  mixste-ttb 3 8 0.1 &

# H173 on GPU 1.
eval_run 1 H173_globalRes_mixsteLoss_lowDrop_frozenH81 \
  "${TRAIN}/H173_globalRes_mixsteLoss_lowDrop_frozenH81/checkpoint_step_0020000.pth" \
  global-residual 4 8 0.1 &

wait
echo "[queue-eval] all done $(date --iso-8601=seconds)"
