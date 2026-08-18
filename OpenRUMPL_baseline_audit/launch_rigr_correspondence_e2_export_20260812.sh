#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Export_20260812
H76_TRAIN=${BASE}/Counterfactual_View_Utility_20260811/train_hypotheses
E2_TRAIN=${BASE}/Learnable_Triangulation_20260814/learned_candidate_cache
H76_VAL=${BASE}/Counterfactual_View_Utility_20260811/H76_validation_all_subsets.npz
E2_VAL=${BASE}/Learnable_Triangulation_20260814/learned_candidate_cache/H76_pairwise_validation.npz
GROUP_IDS=${BASE}/RIGR_P2_feature_export_20260812/balanced20k_group_indices.npy
TOKENS=${BASE}/RIGR_P2_feature_tokens_20260812/balanced20k/subset_tokens.npy
VAL_TOKENS=${BASE}/RIGR_P2_feature_tokens_20260812/val_subset_tokens.npy
TRAIN_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_train.pkl
VAL_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_validation.pkl

mkdir -p "${ROOT}/no_bias_seed0" "${ROOT}/bias_seed1"

run_one() {
  local gpu=$1
  local variant=$2
  local checkpoint=$3
  local extra=()
  local out=${ROOT}/${variant}
  if [[ -s "${out}/manifest.json" && -s "${out}/validation_rigr_e2.npz" ]]; then
    echo "${variant} already exported; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u "${AUDIT}/export_rigr_refined_e2_candidates_20260812.py" \
    --h76-train "${H76_TRAIN}/H76_train_all_subsets_shard0of2.npz" "${H76_TRAIN}/H76_train_all_subsets_shard1of2.npz" \
    --e2-train "${E2_TRAIN}/H76_pairwise_train_shard0of2.npz" "${E2_TRAIN}/H76_pairwise_train_shard1of2.npz" \
    --selected-group-ids "${GROUP_IDS}" --tokens "${TOKENS}" \
    --rigr-checkpoint "${checkpoint}" \
    --validation-h76 "${H76_VAL}" --validation-e2 "${E2_VAL}" \
    --validation-tokens "${VAL_TOKENS}" --output-dir "${out}" \
    --device cuda:0 --batch-size 64 --shards 2 >"${out}/export.log" 2>&1
}

run_one 0 no_bias_seed0 \
  /mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_20260812/correspondence_seed0/model_best.pth &
pid0=$!
run_one 1 bias_seed1 \
  /mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_20260812/correspondence_bias_seed1/model_best.pth &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T RIGR correspondence E2 candidate export completed'
