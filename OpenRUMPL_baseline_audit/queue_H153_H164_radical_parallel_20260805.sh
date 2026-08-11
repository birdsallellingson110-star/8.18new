#!/usr/bin/env bash
# Radical sprint: 2 jobs/GPU, overlaps with H141 queue (intentionally fills VRAM).
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LAUNCH=${AUDIT}/launch_H153_H164_radical_sprint_20260805.sh
LOG=${ROOT}/H153_H164_radical_sprint/queue_parallel.log

mkdir -p "${ROOT}/H153_H164_radical_sprint"
exec >>"${LOG}" 2>&1
echo "======== radical sprint $(date --iso-8601=seconds) ========"

source "${AUDIT}/experiment_should_skip.sh"

run_one() {
  local gpu=$1 var=$2
  echo "[radical] GPU${gpu} ${var} start $(date --iso-8601=seconds)"
  bash "${LAUNCH}" "${var}" "${gpu}" || echo "[radical] WARN ${var} failed"
  echo "[radical] GPU${gpu} ${var} done $(date --iso-8601=seconds)"
}

VARIANTS=(
  h153_skip_vft
  h154_skip_vft_pft
  h155_jv2_skip_vft
  h156_vft1
  h157_no_tri_skip_vft
  h158_tri_only_no_pft
  h159_jv2_skip_vft_pft
  h160_h76_skip_vft
  h161_vft1_skip_pft
  h162_skip_vft_graph
  h163_h76_set_dec
  h164_skip_vft_relview
)

# Wave launch: 4 concurrent (2 per GPU) until all variants scheduled.
i=0
while [[ $i -lt ${#VARIANTS[@]} ]]; do
  batch=()
  while [[ ${#batch[@]} -lt 4 && $i -lt ${#VARIANTS[@]} ]]; do
    batch+=("${VARIANTS[$i]}")
    i=$((i + 1))
  done
  [[ ${#batch[@]} -ge 1 ]] && run_one 0 "${batch[0]}" &
  [[ ${#batch[@]} -ge 2 ]] && run_one 1 "${batch[1]}" &
  [[ ${#batch[@]} -ge 3 ]] && run_one 0 "${batch[2]}" &
  [[ ${#batch[@]} -ge 4 ]] && run_one 1 "${batch[3]}" &
  wait
done

python3 "${AUDIT}/scan_experiment_skip_registry_20260805.py" || true
echo "[radical] finished $(date --iso-8601=seconds)"
