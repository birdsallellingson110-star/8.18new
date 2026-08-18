#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
EXPORT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Export_20260812
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_V234_Cache_20260812

mkdir -p "${ROOT}/no_bias_seed0/train" "${ROOT}/no_bias_seed0/validation" \
         "${ROOT}/bias_seed1/train" "${ROOT}/bias_seed1/validation"

run_one() {
  local variant=$1
  local input=${EXPORT}/${variant}
  local out=${ROOT}/${variant}
  if [[ -s "${out}/validation/manifest.json" ]]; then
    echo "${variant} already built; skip"
    return 0
  fi
  "${PY}" -u "${AUDIT}/build_e2_v234_candidate_cache_20260812.py" \
    --input-files "${input}/train_rigr_e2_shard0of2.npz" "${input}/train_rigr_e2_shard1of2.npz" \
    --output-dir "${out}/train" --prefix train --shards 2 \
    >"${out}/build_train.log" 2>&1
  "${PY}" -u "${AUDIT}/build_e2_v234_candidate_cache_20260812.py" \
    --input-files "${input}/validation_rigr_e2.npz" \
    --output-dir "${out}/validation" --prefix validation --shards 1 \
    >"${out}/build_validation.log" 2>&1
}

run_one no_bias_seed0 &
pid0=$!
run_one bias_seed1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
date '+%F %T correspondence V234 cache build completed'
