#!/usr/bin/env bash
# Restart H147/H148 after nv4 fix (stop stale epoch-6 runs first).
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
LAUNCH="${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh"
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
BASE=${ROOT}/H141_H152_arch_sprint

for tag in \
  H147_H81_vftMask04_w322_ftH81_workers8_seed0_20260805 \
  H148_H81_globalJV1_confgeom_w322_ftH81_workers8_seed0_20260805
do
  rm -f "${BASE}/completed/${tag}.done" "${BASE}/locks/${tag}.lock" \
    "${BASE}/checkpoints/${tag}.txt"
  find /mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999 \
    -maxdepth 1 -type d -name "${tag}_*" -print0 2>/dev/null \
    | xargs -0r rm -rf
done

pkill -f 'exp-name H147_H81_vftMask04_w322' 2>/dev/null || true
pkill -f 'exp-name H148_H81_globalJV1_confgeom_w322' 2>/dev/null || true
sleep 2

export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_EVAL_STRICT=0

bash "${LAUNCH}" h147_vft_mask04 0 &
bash "${LAUNCH}" h148_jv1_biased_h81 1 &
wait
echo "[restart] H147/H148 relaunch done $(date --iso-8601=seconds)"
