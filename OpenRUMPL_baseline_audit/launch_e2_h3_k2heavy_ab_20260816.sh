#!/usr/bin/env bash
# H4/H5: E2 utility on the completed H3 K2-heavy generator.
#
# standard = the audited E2 loss; hinge = the previously registered
# identity-preserving loss (lambda=0.25, V2 multiplier=4).  The candidate
# cache, checkpoint, input protocol and 15E schedule are identical.  Each
# family is run with two seeds; two low-memory jobs share each GPU so the A/B
# can finish together without changing the data or model definition.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT=$(cat /mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/b1_highlr_sampling_ab/K2HEAVY/checkpoint.txt)
TYPE=gbt_yolox_x_score001_fallback_legswap
TRAIN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_input_protocol_v1
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_ab_protocol_v1
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}

mkdir -p "${ROOT}" "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${CKPT}" && test -s "${TRAIN}" && test -s "${VAL}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

{
  echo "experiment=H4_H5_e2_h3_k2heavy_ab"
  echo "started=$(date --iso-8601=seconds)"
  echo "checkpoint=${CKPT}"
  echo "type=${TYPE}"
  sha256sum "${CFG}" "${CKPT}" "${TRAIN}" "${VAL}"
} >"${OUT}/manifest.txt"

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
  if [[ -s "${output}" ]]; then return 0; fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
  export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
  cd "${REPO}"
  "${PY}" -u "${AUDIT}/export_h76_train_subset_hypotheses_20260811.py" \
    --cfg "${CFG}" --checkpoint "${CKPT}" --dataset-name annot_filtered_5_64 \
    --mmpose-type "${TYPE}" --subset "${subset}" --flip-lower-body-kp-test false \
    --output "${output}" --batch-size 256 --workers 8 --gpu 0 \
    >"${output%.npz}.log" 2>&1
}

if [[ ! -s "${ROOT}/train_h3_11c.npz" || ! -s "${ROOT}/validation_h3_11c.npz" ]]; then
  export_cache 0 train "${ROOT}/train_h3_11c.npz" & p0=$!
  export_cache 1 validation "${ROOT}/validation_h3_11c.npz" & p1=$!
  wait "${p0}" "${p1}"
fi

if [[ ! -s "${ROOT}/train_h3_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${ROOT}/train_h3_11c.npz" --output "${ROOT}/train_h3_22c.npz" \
    >"${ROOT}/append_train.log" 2>&1
fi
if [[ ! -s "${ROOT}/validation_h3_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${ROOT}/validation_h3_11c.npz" --output "${ROOT}/validation_h3_22c.npz" \
    >"${ROOT}/append_validation.log" 2>&1
fi
test -s "${ROOT}/train_h3_22c.npz" && test -s "${ROOT}/validation_h3_22c.npz"

run_train() {
  local family="$1" seed="$2" gpu="$3" hinge="$4" v2weight="$5"
  local dir="${OUT}/${family}/seed${seed}"
  if [[ -s "${dir}/result.json" ]]; then return 0; fi
  mkdir -p "${dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" -u "${AUDIT}/train_current_e2_confidence_20260815.py" \
      --train-shards "${ROOT}/train_h3_22c.npz" \
      --validation-cache "${ROOT}/validation_h3_22c.npz" \
      --output-dir "${dir}" --attention-depth 2 \
      --pretrain-epochs 10 --finetune-epochs 5 --batch-size 256 \
      --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
      --identity-hinge "${hinge}" --identity-v2-weight "${v2weight}" \
      --workers 0 --seed "${seed}" --gpu 0 \
      >"${dir}/train.log" 2>&1
  ) &
}

# Two jobs per GPU: each E2 scorer is small enough for the 24 GB cards.
run_train standard 0 0 0.0 1.0
run_train hinge 0 0 0.25 4.0
run_train standard 1 1 0.0 1.0
run_train hinge 1 1 0.25 4.0
wait

# Reuse the pre-registered V2 temperature calibration (0.4) and GHT T=1.8
# for V3/V4.  This is evaluation-only and never uses S9/S11 to select T.
for family in standard hinge; do
  mkdir -p "${OUT}/${family}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/evaluate_e2_c2_calibrated_20260815.py" \
    --cache "${ROOT}/validation_h3_22c.npz" \
    --checkpoint-root "${OUT}/${family}" \
    --output "${OUT}/${family}/calibrated_v2t04.json" \
    --v2-temperature 0.4 --v3-temperature 1.8 --v4-temperature 1.8 \
    --batch-size 1024 --gpu 0 \
    >"${OUT}/${family}/calibration.log" 2>&1
done
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[H4/H5] complete $(date --iso-8601=seconds)"
