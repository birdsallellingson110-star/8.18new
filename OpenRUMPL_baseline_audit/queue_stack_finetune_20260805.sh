#!/usr/bin/env bash
# Re-run multiview ablations as finetune stacks (H76/H81 ckpt), not scratch.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
# shellcheck source=/dev/null
source "${AUDIT}/experiment_should_skip.sh"
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_stack_finetune_20260805.log
L119=${AUDIT}/launch_H119_H126_multiview_v34_20260805.sh
L112=${AUDIT}/launch_H112_H116_beat_gbt_20260805.sh
L129=${AUDIT}/launch_H129_H134_literature_fusion_20260805.sh

mkdir -p "${ROOT}"
exec 9>"${ROOT}/queue_stack_finetune_20260805.lock"
flock -n 9 || { echo "[stack-ft] already running"; exit 0; }
exec >>"${LOG}" 2>&1

echo "======== stack finetune queue $(date --iso-8601=seconds) ========"

pkill -f 'exp-name H124_H76_nestedViewMono_w005_w322_workers8_seed0_20260805' 2>/dev/null || true
pkill -f 'exp-name H127_H81_perJointGate_mono005_w322_workers8_seed0_20260805' 2>/dev/null || true
echo "[stack-ft] stopped scratch H124/H127"

export RUMPL_WORKERS=8

run_bg() {
  local gpu=$1 launch=$2 var=$3
  if experiment_should_skip_variant "${var}" 2>/dev/null; then
    echo "[stack-ft] skip ${var} (skip registry alias)"
    return 0
  fi
  echo "[stack-ft] GPU${gpu} ${launch##*/} ${var}"
  bash "${launch}" "${var}" "${gpu}" &
}

# Phase 1: two GPUs
run_bg 0 "${L112}" h113_reproj
run_bg 1 "${L119}" h127_mono_h81
wait

# Phase 2
run_bg 0 "${L119}" h128_relview_h81
run_bg 1 "${L119}" h123_gjv2
wait

# Phase 3
run_bg 0 "${L119}" h124_mono
run_bg 1 "${L129}" h130_relview_mask05
wait

run_bg 0 "${L129}" h132_gjv1_biased
wait

echo "[stack-ft] done $(date --iso-8601=seconds)"
