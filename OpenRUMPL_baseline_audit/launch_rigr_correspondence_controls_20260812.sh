#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_20260812
TRAIN_CACHE=${BASE}/Counterfactual_View_Utility_20260811/train_hypotheses
VAL_CACHE=${BASE}/Counterfactual_View_Utility_20260811/H76_validation_all_subsets.npz
TOKENS=${BASE}/RIGR_P2_feature_tokens_20260812/balanced20k/subset_tokens.npy
VAL_TOKENS=${BASE}/RIGR_P2_feature_tokens_20260812/val_subset_tokens.npy
GROUP_IDS=${BASE}/RIGR_P2_feature_export_20260812/balanced20k_group_indices.npy
TRAIN_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_train.pkl
VAL_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_validation.pkl

mkdir -p "${ROOT}/correspondence_bias_seed0" "${ROOT}/correspondence_seed1"

run_one() {
  local physical_gpu=$1
  local seed=$2
  local variant=$3
  local out=${ROOT}/${variant}
  if [[ -s "${out}/result.json" ]]; then
    echo "${variant} already has result.json; skip"
    return 0
  fi
  local extra=()
  if [[ "${variant}" == "correspondence_bias_seed0" ]]; then
    extra+=(--geometry-biased-attention)
  fi
  CUDA_VISIBLE_DEVICES=${physical_gpu} "${PY}" -u "${AUDIT}/train_rigr_hrnet_feature_20260812.py" \
    --train-cache "${TRAIN_CACHE}/H76_train_all_subsets_shard0of2.npz" "${TRAIN_CACHE}/H76_train_all_subsets_shard1of2.npz" \
    --validation-cache "${VAL_CACHE}" \
    --train-pkl "${TRAIN_PKL}" --validation-pkl "${VAL_PKL}" \
    --train-tokens "${TOKENS}" --validation-tokens "${VAL_TOKENS}" \
    --output-dir "${out}" --device cuda:0 --epochs 8 --batch-size 256 --workers 0 \
    --lr 3e-4 --weight-decay 1e-4 --max-train-groups 10000 \
    --train-group-indices-file "${GROUP_IDS}" --holdout-every 10 --seed "${seed}" \
    --correspondence-attention "${extra[@]}" >"${out}/train.log" 2>&1
}

run_one 0 0 correspondence_bias_seed0 &
pid0=$!
run_one 1 1 correspondence_seed1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T RIGR cross-view correspondence controls completed'
