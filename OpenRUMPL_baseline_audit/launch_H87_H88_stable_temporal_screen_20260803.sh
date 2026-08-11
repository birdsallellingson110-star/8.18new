#!/usr/bin/env bash
set -euo pipefail
physical_gpu=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/H87_H88_stable_temporal_screen
INPUT_DONE=${ROOT}/H84_temporal_stride5_validation_inputs/completed.done
mkdir -p "${OUT}/logs" "${OUT}/H87_unbiased" "${OUT}/H88_biased" "${OUT}/eval"
exec 9>"${OUT}/pipeline.lock"
flock 9
while [[ ! -s "${INPUT_DONE}" ]]; do sleep 30; done
export CUDA_VISIBLE_DEVICES="${physical_gpu}" PYTHONPATH="${REPO}/lib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
common=(--cfg "${CFG}" --base-checkpoint "${BASE}" --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64 --window-length 9 --frame-stride 5 --num-views 2 --depth 3 --heads 8 --token-dropout 0.2 --optimizer-steps 500 --warmup-steps 50 --micro-batch-size 1 --effective-batch-size 8 --workers 4 --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 --amp-dtype bf16 --log-every 10 --save-every 500)
"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" "${common[@]}" --output-dir "${OUT}/H87_unbiased" >"${OUT}/logs/H87_train.log" 2>&1 & p0=$!
"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" "${common[@]}" --biased --output-dir "${OUT}/H88_biased" >"${OUT}/logs/H88_train.log" 2>&1 & p1=$!
wait "${p0}"; wait "${p1}"
H87=${OUT}/H87_unbiased/checkpoint_step_0000500.pth
H88=${OUT}/H88_biased/checkpoint_step_0000500.pth
test -s "${H87}"; test -s "${H88}"
eval_one() {
  local name=$1 ckpt=$2 biased=$3
  for v in 2 4; do
    d=${OUT}/eval/${name}/V${v}; mkdir -p "${d}"
    extra=(); [[ -n "${ckpt}" ]] && extra+=(--temporal-checkpoint "${ckpt}"); [[ "${biased}" == 1 ]] && extra+=(--biased)
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" --cfg "${CFG}" --base-checkpoint "${BASE}" "${extra[@]}" --output-dir "${d}" --mmpose-type "${TYPE}" --dataset-name annot_temporal_5_5 --num-views "${v}" --window-length 9 --frame-stride 5 --depth 3 --heads 8 --batch-size 8 --workers 6 --device cuda:0 >"${d}/eval.log" 2>&1
  done
}
eval_one H76_gate0 "" 0
eval_one H87_unbiased "${H87}" 0
eval_one H88_biased "${H88}" 1
date --iso-8601=seconds >"${OUT}/completed.done"
echo "[H87-H88] complete $(date --iso-8601=seconds)" | tee "${OUT}/pipeline.log"
