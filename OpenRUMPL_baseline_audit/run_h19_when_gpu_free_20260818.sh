#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPU=${1:?usage: run_h19_when_gpu_free_20260818.sh PHYSICAL_GPU VARIANT...}
shift
case "${PHYSICAL_GPU}" in
  0|1) ;;
  *) echo "physical GPU must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$#" -lt 1 ]]; then
  echo "at least one variant is required" >&2
  exit 2
fi

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
while [[ "$#" -gt 0 ]]; do
  VARIANT=$1
  shift
  while true; do
    USED=$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ "${USED}" =~ ^[0-9]+$ ]] && (( USED < 4096 )); then
      break
    fi
    sleep 45
  done
  "${AUDIT}/launch_h19_causal_seq2seq_variant_20260818.sh" \
    "${VARIANT}" "${PHYSICAL_GPU}"
done
