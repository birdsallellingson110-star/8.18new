#!/usr/bin/env bash
# H117/H118: H40-style global JVT on H76 (HRNet→A1D→H21), not MixSTE.
# Goal: accuracy only — T=9 causal-latest eval on V2/V3/V4.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {h117_frozen_latest|h118_unfreeze_vft_latest} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
INPUT_DONE=${ROOT}/H84_temporal_stride5_validation_inputs/completed.done
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
OUT=${ROOT}/H117_H118_temporal_jvt_accuracy

mkdir -p "${OUT}/logs" "${OUT}/eval"
exec 9>"${OUT}/pipeline.lock"
flock 9
test -s "${INPUT_DONE}"
test -s "${BASE}"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${REPO}/lib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

common=(
  --cfg "${CFG}" --base-checkpoint "${BASE}"
  --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64
  --fusion-mode global-residual
  --window-length 9 --frame-stride 5 --num-views 2
  --depth 4 --heads 8 --token-dropout 0.2
  --optimizer-steps 6000 --warmup-steps 600
  --micro-batch-size 1 --effective-batch-size 8 --workers 4
  --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0
  --amp-dtype bf16 --log-every 50 --save-every 2000
  --disable-missing-keypoints --loss-type mpjpe --loss-frame latest
)

case "${variant}" in
  h117_frozen_latest)
    name=H117_jvt_depth4_latest_frozen_biased
    extra=(--biased)
    train_extra=()
    ;;
  h118_unfreeze_vft_latest)
    name=H118_jvt_depth4_latest_unfreeze_vft_biased
    extra=(--biased)
    train_extra=(
      --unfreeze-backbone --backbone-train-scope vft
      --backbone-lr-multiplier 0.1 --backbone-eval-mode
    )
    ;;
  *)
    echo "unsupported: ${variant}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT}/${name}"
echo "[${name}] train $(date --iso-8601=seconds)" | tee "${OUT}/logs/${name}.log"
"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
  "${common[@]}" "${train_extra[@]}" "${extra[@]}" \
  --output-dir "${OUT}/${name}" >>"${OUT}/logs/${name}.log" 2>&1

ckpt=${OUT}/${name}/checkpoint_step_0006000.pth
test -s "${ckpt}"

for views in 2 3 4; do
  dest=${OUT}/eval/${name}/V${views}
  mkdir -p "${dest}"
  "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
    --cfg "${CFG}" --base-checkpoint "${BASE}" \
    --temporal-checkpoint "${ckpt}" --biased \
    --output-dir "${dest}" --mmpose-type "${TYPE}" \
    --dataset-name annot_temporal_5_5 --num-views "${views}" \
    --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
    --batch-size 32 --workers 6 --device cuda:0 \
    >"${dest}/eval.log" 2>&1
done

date --iso-8601=seconds >"${OUT}/completed_${name}.done"
echo "[${name}] done $(date --iso-8601=seconds)"
