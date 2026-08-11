#!/usr/bin/env bash
# Retry failed H141–H146 with N_VIEWS=4 fix (via launch_H59 defaults). H147/H148 may need manual restart if still on old code.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
LAUNCH="${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh"
LOG_DIR=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H141_H152_arch_sprint/logs
mkdir -p "${LOG_DIR}"

variants=(
  h141_no_pft_repeat
  h142_relview_w322
  h143_gjv2_h81
  h144_graph_res_h81
  h145_no_tri_anchor
  h146_set_decoder_h76
)

run_one() {
  local gpu=$1 variant=$2
  local log="${LOG_DIR}/retry_nv4_${variant}_gpu${gpu}.log"
  echo "[retry] gpu=${gpu} variant=${variant} $(date --iso-8601=seconds)" | tee -a "${log}"
  bash "${LAUNCH}" "${variant}" "${gpu}" >>"${log}" 2>&1 \
    && echo "[ok] ${variant}" | tee -a "${log}" \
    || echo "[warn] failed ${variant}" | tee -a "${log}"
}

export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_EVAL_STRICT=0

MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H141_H152_arch_sprint

clear_partial_train() {
  local prefix=$1
  find "${MODEL_OUTPUT}" -maxdepth 1 -type d -name "${prefix}_*" -print0 2>/dev/null \
    | xargs -0r rm -rf
  rm -f "${BASE}/checkpoints/${prefix}.txt" "${BASE}/completed/${prefix}.done"
}

for prefix in \
  H141_H81_noPftRepeatLast_w322_ftH81_workers8_seed0_20260805 \
  H142_H81_relViewFusion_w322_ftH81_workers8_seed0_20260805 \
  H143_H81_globalJV2_rezero_w322_ftH81_workers8_seed0_20260805 \
  H144_H81_postPftGraphRes_w322_ftH81_workers8_seed0_20260805 \
  H145_H81_noTriAnchor_w322_ftH81_workers8_seed0_20260805 \
  H146_H76_gbtSetDecoder_w322_ftH76_workers8_seed0_20260805
do
  clear_partial_train "${prefix}"
done

# 2-wide on GPUs 0 and 1
for i in "${!variants[@]}"; do
  gpu=$((i % 2))
  run_one "${gpu}" "${variants[$i]}" &
  if (( (i + 1) % 2 == 0 )); then
    wait
  fi
done
wait
echo "[queue] H141-H146 retry done $(date --iso-8601=seconds)"
