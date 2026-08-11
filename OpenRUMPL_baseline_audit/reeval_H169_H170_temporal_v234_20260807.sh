#!/usr/bin/env bash
# Re-eval temporal checkpoints with C(4,k) grouping per --num-views (Table-II aligned).
set -euo pipefail

gpu=${1:-0}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
H81=$(tr -d '\r\n' < "${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt")
TRAIN=${ROOT}/H169_H174_temporal_h81_gbt_aligned
EVAL=${TRAIN}/eval
LOG=${TRAIN}/logs/reeval_H169_H170_v234.log

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${REPO}/lib"
export RUMPL_EVAL_STRICT=0
unset RUMPL_N_VIEWS_TRAIN_TEST_ALL

mkdir -p "${TRAIN}/logs"
exec >>"${LOG}" 2>&1
echo "[reeval] GPU${gpu} start $(date --iso-8601=seconds)"

reeval_one() {
  local name=$1 ckpt=$2 fusion=$3 depth=$4 heads=$5 rscale=$6
  test -s "${ckpt}"
  local args=(--fusion-mode "${fusion}" --depth "${depth}" --heads "${heads}" --residual-scale "${rscale}" --biased)
  for v in 2 3 4; do
    local dest="${EVAL}/${name}/V${v}"
    mkdir -p "${dest}"
    echo "[reeval] ${name} V${v} $(date --iso-8601=seconds)"
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
      --cfg "${CFG}" --base-checkpoint "${H81}" --backbone-flavor h81 \
      --temporal-checkpoint "${ckpt}" "${args[@]}" \
      --output-dir "${dest}" --mmpose-type "${TYPE}" \
      --dataset-name annot_temporal_5_5 --num-views "${v}" \
      --window-length 9 --frame-stride 5 \
      --batch-size 16 --workers 6 --device cuda:0 \
      >"${dest}/eval.log" 2>&1
    "${PY}" -c "import json; d=json.load(open('${dest}/table2.json')); print('  ->', d['windows'], round(d['table2_action_equal']['all17_mm'],2))"
  done
}

reeval_one H169_mixsteTTB_mixsteLoss_frozenH81 \
  "${TRAIN}/H169_mixsteTTB_mixsteLoss_frozenH81/checkpoint_step_0020000.pth" \
  mixste-ttb 3 8 0.1

reeval_one H170_mixsteTTBres_mixsteLoss_frozenH81 \
  "${TRAIN}/H170_mixsteTTBres_mixsteLoss_frozenH81/checkpoint_step_0020000.pth" \
  mixste-ttb-residual 3 8 0.1

echo "[reeval] done $(date --iso-8601=seconds)"
