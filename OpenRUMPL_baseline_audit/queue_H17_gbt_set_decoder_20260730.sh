#!/usr/bin/env bash
set -euo pipefail

HERE=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/h36m_gbt_set_decoder_20260730
mkdir -p "${BASE}/queue"

# GPU1 is free now: start the architecture and Pluecker arms immediately.
(
  "${HERE}/launch_H17_gbt_set_decoder_20260730.sh" set 1
) >"${BASE}/queue/S0_gpu1.log" 2>&1 &
s0_pid=$!
(
  "${HERE}/launch_H17_gbt_set_decoder_20260730.sh" plucker 1
) >"${BASE}/queue/S1_gpu1.log" 2>&1 &
s1_pid=$!

# GPU0 is still completing H16.  Do not interrupt it; begin the two bias
# ablations as soon as the complete H16 queue exits.
(
  while kill -0 1202502 2>/dev/null; do
    sleep 15
  done
  "${HERE}/launch_H17_gbt_set_decoder_20260730.sh" biased 0
) >"${BASE}/queue/S2_gpu0_after_H16.log" 2>&1 &
s2_pid=$!
(
  while kill -0 1202502 2>/dev/null; do
    sleep 15
  done
  "${HERE}/launch_H17_gbt_set_decoder_20260730.sh" full 0
) >"${BASE}/queue/S3_gpu0_after_H16.log" 2>&1 &
s3_pid=$!

printf '%s\n' "${s0_pid}" >"${BASE}/queue/S0_gpu1.pid"
printf '%s\n' "${s1_pid}" >"${BASE}/queue/S1_gpu1.pid"
printf '%s\n' "${s2_pid}" >"${BASE}/queue/S2_gpu0_after_H16.pid"
printf '%s\n' "${s3_pid}" >"${BASE}/queue/S3_gpu0_after_H16.pid"
echo "[H17_QUEUE] S0=${s0_pid} S1=${s1_pid} S2_wait=${s2_pid} S3_wait=${s3_pid}"
wait
