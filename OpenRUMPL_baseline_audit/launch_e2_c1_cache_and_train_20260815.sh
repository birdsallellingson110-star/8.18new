#!/usr/bin/env bash
# E2-C1: apply the audited utility scorer to the single C1 checkpoint.  C1 is
# one H76/RUMPL model (B2 -> mixed-cardinality recovery); this experiment does
# not splice predictions from different checkpoints.  It tests whether the
# utility readout can retain C1's V2 gain while recovering E2-C2's V3/V4 gain.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C1_B2_TO_MIXED_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-03-00/model_best.pth.tar
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c1_input_protocol_v1
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c1_training_protocol_v1
TYPE=gbt_yolox_x_score001_fallback_legswap
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
TRAIN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl

mkdir -p "${ROOT}" "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${CKPT}" && test -s "${TRAIN}" && test -s "${VAL}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

link_split() {
  local split="$1" source="$2"
  local target="${TYPE_DIR}/h36m_${split}.pkl"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      echo "mismatched dataset link ${target}" >&2; exit 2;
    }
  else
    ln -s "${source}" "${target}"
  fi
}
link_split train "${TRAIN}"
link_split validation "${VAL}"

export_cache() {
  local gpu="$1" subset="$2" output="$3"
  if [[ -s "${output}" ]]; then
    echo "[E2-C1] ${subset} cache already exists"
    return 0
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
  cd "${REPO}"
  "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" \
    --dataset-name annot_filtered_5_64 \
    --mmpose-type "${TYPE}" --subset "${subset}" \
    --flip-lower-body-kp-test false --output "${output}" \
    --batch-size 256 --workers 8 --gpu 0 \
    >"${output%.npz}.log" 2>&1
}

if [[ ! -s "${ROOT}/train_c1_11c.npz" || ! -s "${ROOT}/validation_c1_11c.npz" ]]; then
  export_cache 0 train "${ROOT}/train_c1_11c.npz" & p0=$!
  export_cache 1 validation "${ROOT}/validation_c1_11c.npz" & p1=$!
  wait "${p0}" "${p1}"
fi

if [[ ! -s "${ROOT}/train_c1_22c.npz" || ! -s "${ROOT}/validation_c1_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${ROOT}/train_c1_11c.npz" \
    --output "${ROOT}/train_c1_22c.npz" \
    >"${ROOT}/append_confidence_train.log" 2>&1
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${ROOT}/validation_c1_11c.npz" \
    --output "${ROOT}/validation_c1_22c.npz" \
    >"${ROOT}/append_confidence_validation.log" 2>&1
fi

run_train() {
  local seed="$1" gpu="$2"
  local dir="${OUT}/seed${seed}"
  if [[ -s "${dir}/result.json" ]]; then return 0; fi
  mkdir -p "${dir}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
    --train-shards "${ROOT}/train_c1_22c.npz" \
    --validation-cache "${ROOT}/validation_c1_22c.npz" \
    --output-dir "${dir}" \
    --pretrain-epochs 10 --finetune-epochs 5 \
    --batch-size 256 --temperature 1.8 --target-temperature-mm 5.0 \
    --oracle-weight 1.0 --workers 0 --seed "${seed}" --gpu 0 \
    >"${dir}/train.log" 2>&1
}

run_train 0 0 & p0=$!
run_train 1 1 & p1=$!
wait "${p0}" "${p1}"
echo "[E2-C1] complete $(date --iso-8601=seconds)"
