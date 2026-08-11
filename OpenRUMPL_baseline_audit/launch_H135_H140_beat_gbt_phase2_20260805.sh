#!/usr/bin/env bash
# H135-H140: beat GBT-HRNet Table-I (H36M V2/V3/V4 All-17) + CMU cross-dataset follow-ups.
# H81 single-frame already: V2 ok, V3 ~+0.4mm, V4 ~+4mm vs GBT 9-frame reference.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {h135_temporal_h81|h136_temporal_h81_unfreeze|h137_h81_depro_caa|h138_h114_v4w|h139_h117_temporal_h76|h140_h81_gbt_bias_ft} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2

if experiment_should_skip_variant "${variant}" 2>/dev/null; then
  echo "[${variant}] skip (alias target already complete — see EXPERIMENT_SKIP_REGISTRY_20260805.md)"
  exit 0
fi

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
H76=${ROOT}/H76_h50_centered_plucker/checkpoints/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803.txt
H81=$(tr -d '\r\n' < "${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt")
INPUT_DONE=${ROOT}/H84_temporal_stride5_validation_inputs/completed.done
OUT=${ROOT}/H135_H140_beat_gbt_phase2
# shellcheck source=/dev/null
source "${AUDIT}/experiment_should_skip.sh"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${REPO}/lib"

temporal_train_eval() {
  local name=$1 base_ckpt=$2 flavor=$3 extra_train=("${@:4}")
  local log="${OUT}/logs/${name}.log"
  mkdir -p "${OUT}/${name}" "${OUT}/logs" "${OUT}/eval/${name}"
  if [[ -s "${OUT}/completed_${name}.done" ]]; then
    echo "[${name}] skip completed" | tee -a "${log}"
    return 0
  fi
  test -s "${INPUT_DONE}"
  test -s "${base_ckpt}"
  echo "[${name}] train $(date --iso-8601=seconds)" | tee "${log}"
  "${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" \
    --cfg "${CFG}" --base-checkpoint "${base_ckpt}" \
    --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64 \
    --fusion-mode global-residual --backbone-flavor "${flavor}" \
    --window-length 9 --frame-stride 5 --num-views 2 \
    --depth 4 --heads 8 --token-dropout 0.2 --biased \
    --optimizer-steps 6000 --warmup-steps 600 \
    --micro-batch-size 1 --effective-batch-size 8 --workers 4 \
    --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 \
    --amp-dtype bf16 --log-every 50 --save-every 2000 \
    --disable-missing-keypoints --loss-type mpjpe --loss-frame latest \
    --output-dir "${OUT}/${name}" "${extra_train[@]}" >>"${log}" 2>&1
  local ckpt="${OUT}/${name}/checkpoint_step_0006000.pth"
  test -s "${ckpt}"
  for views in 2 3 4; do
    local dest="${OUT}/eval/${name}/V${views}"
    mkdir -p "${dest}"
    "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" \
      --cfg "${CFG}" --base-checkpoint "${base_ckpt}" \
      --temporal-checkpoint "${ckpt}" --biased --backbone-flavor "${flavor}" \
      --output-dir "${dest}" --mmpose-type "${TYPE}" \
      --dataset-name annot_temporal_5_5 --num-views "${views}" \
      --window-length 9 --frame-stride 5 --depth 4 --heads 8 \
      --batch-size 32 --workers 6 --device cuda:0 \
      >"${dest}/eval.log" 2>&1
  done
  date --iso-8601=seconds >"${OUT}/completed_${name}.done"
}

case "${variant}" in
  h135_temporal_h81)
    temporal_train_eval \
      H135_temporal_h81_T9_biased_frozen "${H81}" h81
    ;;
  h136_temporal_h81_unfreeze)
    temporal_train_eval \
      H136_temporal_h81_T9_unfreeze_vft "${H81}" h81 \
      --unfreeze-backbone --backbone-train-scope vft \
      --backbone-lr-multiplier 0.1 --backbone-eval-mode
    ;;
  h137_h81_depro_caa)
    export CODE_OVERRIDE=H137
    export TAG_OVERRIDE=H137_H81_depro01_caa01_ftH81_workers8_seed0_20260805
    export BASE_OVERRIDE=${OUT}
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 + DePro ray reliability + CAA conf fusion (GBT-like)"
    export DEPRO_LAMBDA=0.1
    export CAA_LAMBDA=0.1
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_WORKERS=8
    exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
    ;;
  h138_h114_v4w)
    exec bash "${AUDIT}/launch_H112_H116_beat_gbt_20260805.sh" h114_v4_train_weight "${physical_gpu}"
    ;;
  h139_h117_temporal_h76)
    exec bash "${AUDIT}/launch_H117_H118_temporal_jvt_accuracy_20260805.sh" \
      h117_frozen_latest "${physical_gpu}"
    ;;
  h140_h81_gbt_bias_ft)
    export CODE_OVERRIDE=H140
    export TAG_OVERRIDE=H140_H81_VFT_gbtBias_ftH81_workers8_seed0_20260805
    export BASE_OVERRIDE=${OUT}
    export RUMPL_STACK_FROM=H81
    export CONTROL_NOTE_OVERRIDE="H81 finetune + GBT conf/geom bias (H112 regressed on H76; retry on H81)"
    export GBT_LEARNABLE_BIAS=1
    export GBT_USE_CONF_BIAS=1
    export GBT_USE_GEOM_BIAS=1
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_WORKERS=8
    exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
    ;;
  *)
    echo "unsupported: ${variant}" >&2
    exit 2
    ;;
esac

printf '[%s] pipeline done %s\n' "${variant}" "$(date --iso-8601=seconds)"
