#!/usr/bin/env bash
# Paired temporal screen: identical frozen H76 backbone, without/with GBT bias.
set -euo pipefail

physical_gpu=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
INPUT_DONE=${ROOT}/H84_temporal_stride5_validation_inputs/completed.done
OUT=${ROOT}/H85_H86_h76_temporal_screen
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap

mkdir -p "${OUT}/logs" "${OUT}/H85_unbiased" "${OUT}/H86_biased" "${OUT}/eval"
exec 9>"${OUT}/pipeline.lock"
flock 9
while [[ ! -s "${INPUT_DONE}" ]]; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${REPO}/lib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "[H85-H86] start $(date --iso-8601=seconds)" | tee "${OUT}/pipeline.log"

common=(
  --cfg "${CFG}" --base-checkpoint "${BASE_CKPT}"
  --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64
  --window-length 9 --frame-stride 5 --num-views 2
  --depth 3 --heads 8 --token-dropout 0.2
  --optimizer-steps 500 --warmup-steps 50
  --micro-batch-size 1 --effective-batch-size 8 --workers 4
  --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0
  --amp-dtype bf16 --log-every 10 --save-every 500
)

"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
  "${common[@]}" --output-dir "${OUT}/H85_unbiased" \
  >"${OUT}/logs/H85_train.log" 2>&1 &
pid0=$!
"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
  "${common[@]}" --biased --output-dir "${OUT}/H86_biased" \
  >"${OUT}/logs/H86_train.log" 2>&1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

H85=${OUT}/H85_unbiased/checkpoint_step_0000500.pth
H86=${OUT}/H86_biased/checkpoint_step_0000500.pth
test -s "${H85}"
test -s "${H86}"

evaluate_one() {
  local name=$1
  local checkpoint=$2
  local bias_flag=$3
  for views in 2 4; do
    destination=${OUT}/eval/${name}/V${views}
    mkdir -p "${destination}"
    temporal_args=()
    if [[ -n "${checkpoint}" ]]; then
      temporal_args+=(--temporal-checkpoint "${checkpoint}")
    fi
    if [[ "${bias_flag}" == 1 ]]; then
      temporal_args+=(--biased)
    fi
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
      --cfg "${CFG}" --base-checkpoint "${BASE_CKPT}" \
      "${temporal_args[@]}" --output-dir "${destination}" \
      --mmpose-type "${TYPE}" --dataset-name annot_temporal_5_5 \
      --num-views "${views}" --window-length 9 --frame-stride 5 \
      --depth 3 --heads 8 --batch-size 8 --workers 6 --device cuda:0 \
      >"${destination}/eval.log" 2>&1
  done
}

evaluate_one H76_gate0 "" 0
evaluate_one H85_unbiased "${H85}" 0
evaluate_one H86_biased "${H86}" 1

date --iso-8601=seconds >"${OUT}/completed.done"
echo "[H85-H86] complete $(date --iso-8601=seconds)" | tee -a "${OUT}/pipeline.log"
