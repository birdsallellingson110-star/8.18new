#!/usr/bin/env bash
# H169-H174: GBT/MixSTE-aligned temporal finetune from H81 (best single-frame stack).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 VARIANT PHYSICAL_GPU" >&2
  echo "  h169_mixste_ttb_frozen h170_mixste_ttb_res h171_mixste_alt h172_mixste_unfreeze_vft" >&2
  echo "  h173_global_mixste_loss h174_mixste_ttb_v4train" >&2
  exit 2
fi

variant=$1
physical_gpu=$2

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
H81=$(tr -d '\r\n' < "${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt")
INPUT_DONE=${ROOT}/H84_temporal_stride5_validation_inputs/completed.done
OUT=${ROOT}/H169_H174_temporal_h81_gbt_aligned

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${REPO}/lib"
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_EVAL_STRICT=0

mkdir -p "${OUT}/logs"

temporal_run() {
  local name=$1
  shift
  local extra_train=("$@")
  local log="${OUT}/logs/${name}.log"
  local done="${OUT}/completed_${name}.done"
  if [[ -s "${done}" ]]; then
    echo "[${name}] skip completed" | tee -a "${log}"
    return 0
  fi
  test -s "${INPUT_DONE}"
  test -s "${H81}"
  echo "[${name}] train $(date --iso-8601=seconds)" | tee "${log}"
  "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
    --cfg "${CFG}" --base-checkpoint "${H81}" \
    --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64 \
    --backbone-flavor h81 --disable-missing-keypoints \
    --window-length 9 --frame-stride 5 \
    --biased --token-dropout 0.1 \
    --optimizer-steps 20000 --warmup-steps 2000 \
    --micro-batch-size 1 --effective-batch-size 16 --workers 6 \
    --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 \
    --amp-dtype bf16 --log-every 100 --save-every 5000 \
    --loss-profile mixste-original --loss-frame all \
    --output-dir "${OUT}/${name}" \
    "${extra_train[@]}" >>"${log}" 2>&1
  local ckpt="${OUT}/${name}/checkpoint_step_0020000.pth"
  test -s "${ckpt}"
  read -r fusion_mode depth heads residual_scale <<<"$(
    "${PY}" -c "import json; a=json.load(open('${OUT}/${name}/run_args.json')); print(a['fusion_mode'], a['depth'], a['heads'], a['residual_scale'])"
  )"
  for views in 2 3 4; do
    local dest="${OUT}/eval/${name}/V${views}"
    mkdir -p "${dest}"
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
      --cfg "${CFG}" --base-checkpoint "${H81}" \
      --temporal-checkpoint "${ckpt}" --biased --backbone-flavor h81 \
      --output-dir "${dest}" --mmpose-type "${TYPE}" \
      --dataset-name annot_temporal_5_5 --num-views "${views}" \
      --window-length 9 --frame-stride 5 \
      --fusion-mode "${fusion_mode}" --depth "${depth}" --heads "${heads}" \
      --residual-scale "${residual_scale}" \
      --batch-size 16 --workers 6 --device cuda:0 \
      >"${dest}/eval.log" 2>&1
    pred=$(find "${dest}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    if [[ -n "${pred}" ]]; then
      "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
        --output-json "${dest}/table2.json" >"${dest}/table2.log" 2>&1 || true
    fi
  done
  date --iso-8601=seconds >"${done}"
}

case "${variant}" in
  h169_mixste_ttb_frozen)
    temporal_run H169_mixsteTTB_mixsteLoss_frozenH81 \
      --fusion-mode mixste-ttb --depth 3 --heads 8 --num-views 2 \
      --residual-scale 0.1
    ;;
  h170_mixste_ttb_res)
    temporal_run H170_mixsteTTBres_mixsteLoss_frozenH81 \
      --fusion-mode mixste-ttb-residual --depth 3 --heads 8 --num-views 2 \
      --residual-scale 0.1
    ;;
  h171_mixste_alt)
    temporal_run H171_mixsteAlt_mixsteLoss_frozenH81 \
      --fusion-mode mixste-alternating --depth 4 --heads 8 --num-views 2 \
      --residual-scale 0.1
    ;;
  h172_mixste_unfreeze_vft)
    temporal_run H172_mixsteTTB_unfreezeVFT_mixsteLoss \
      --fusion-mode mixste-ttb --depth 3 --heads 8 --num-views 2 \
      --unfreeze-backbone --backbone-train-scope vft \
      --backbone-lr-multiplier 0.05 --backbone-eval-mode \
      --residual-scale 0.1
    ;;
  h173_global_mixste_loss)
    temporal_run H173_globalRes_mixsteLoss_lowDrop_frozenH81 \
      --fusion-mode global-residual --depth 4 --heads 8 --num-views 2 \
      --token-dropout 0.05 \
      --residual-scale 0.1
    ;;
  h174_mixste_ttb_v4train)
    temporal_run H174_mixsteTTB_mixsteLoss_trainV4 \
      --fusion-mode mixste-ttb --depth 3 --heads 8 --num-views 4 \
      --residual-scale 0.1
    ;;
  *)
    echo "unknown variant ${variant}" >&2
    exit 2
    ;;
esac

echo "[${variant}] finished $(date --iso-8601=seconds)"
