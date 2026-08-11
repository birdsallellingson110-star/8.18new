#!/usr/bin/env bash
set -euo pipefail
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
LOG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H165_H168_nv4_w322/queue.log
mkdir -p "$(dirname "${LOG}")"
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_EVAL_STRICT=0
for pair in "0 h165_tri_only" "1 h166_vft1_tri"; do
  set -- ${pair}
  gpu=$1; var=$2
  echo "[H165+] gpu=${gpu} ${var} $(date --iso-8601=seconds)" | tee -a "${LOG}"
  bash "${AUDIT}/launch_H165_H168_nv4_w322_20260806.sh" "${gpu}" "${var}" >>"${LOG}" 2>&1 &
done
wait
for pair in "0 h167_shallow_no_tri" "1 h168_gate_baseline"; do
  set -- ${pair}
  gpu=$1; var=$2
  echo "[H165+] gpu=${gpu} ${var} $(date --iso-8601=seconds)" | tee -a "${LOG}"
  bash "${AUDIT}/launch_H165_H168_nv4_w322_20260806.sh" "${gpu}" "${var}" >>"${LOG}" 2>&1 &
done
wait
echo "[H165+] queue done $(date --iso-8601=seconds)" | tee -a "${LOG}"
