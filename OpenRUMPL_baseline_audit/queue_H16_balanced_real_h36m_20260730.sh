#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
HERE=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/h36m_balanced_views_20260730

mkdir -p "${BASE}/queue"

# Put both architecture-clean baselines first.  Their paired bias ablations
# start only after the corresponding baseline is fully trained and evaluated.
(
  "${HERE}/launch_H16_balanced_real_h36m_20260730.sh" random2to4 baseline "${GPU}"
  "${HERE}/launch_H16_balanced_real_h36m_20260730.sh" random2to4 both "${GPU}"
) >"${BASE}/queue/UK_gpu${GPU}.log" 2>&1 &
uk_pid=$!

(
  "${HERE}/launch_H16_balanced_real_h36m_20260730.sh" fixed2 baseline "${GPU}"
  "${HERE}/launch_H16_balanced_real_h36m_20260730.sh" fixed2 both "${GPU}"
) >"${BASE}/queue/F2_gpu${GPU}.log" 2>&1 &
f2_pid=$!

printf '%s\n' "${uk_pid}" >"${BASE}/queue/UK_gpu${GPU}.pid"
printf '%s\n' "${f2_pid}" >"${BASE}/queue/F2_gpu${GPU}.pid"
echo "[H16_QUEUE] gpu=${GPU} UK_pid=${uk_pid} F2_pid=${f2_pid}"
wait
