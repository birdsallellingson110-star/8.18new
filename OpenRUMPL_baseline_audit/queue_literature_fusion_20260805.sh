#!/usr/bin/env bash
# Deduped literature queue — see EXPERIMENT_DEDUP_REGISTRY_20260805.md
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_literature_fusion_20260805.log
LAUNCH=${AUDIT}/launch_H129_H134_literature_fusion_20260805.sh
REG=${ROOT}/EXPERIMENT_DEDUP_REGISTRY_20260805.md

mkdir -p "${ROOT}"
exec 9>"${ROOT}/queue_literature_fusion_20260805.lock"
flock -n 9 || { echo "[lit-dedup] queue already running"; exit 0; }
exec >>"${LOG}" 2>&1

echo "======== deduped literature queue $(date --iso-8601=seconds) ========"

skip_train() {
  local tag=$1
  if [[ -s "${ROOT}/H112_H116_beat_gbt/completed/${tag}.done" ]] \
    || [[ -s "${ROOT}/H119_H126_multiview_v34/completed/${tag}.done" ]] \
    || [[ -s "${ROOT}/H129_H134_literature_fusion/completed/${tag}.done" ]]; then
    echo "[lit-dedup] skip ${tag} (completed)"
    return 0
  fi
  if pgrep -af train_rumpl | grep -q "${tag}"; then
    echo "[lit-dedup] skip ${tag} (already training)"
    return 0
  fi
  return 1
}

run_one() {
  local gpu=$1 var=$2 tag=$3
  if skip_train "${tag}"; then
    return 0
  fi
  export RUMPL_WORKERS=6
  echo "[lit-dedup] GPU${gpu} ${var} tag=${tag} $(date --iso-8601=seconds)"
  bash "${LAUNCH}" "${var}" "${gpu}"
}

# Stop duplicates per registry
pkill -f 'H122_H76_MTF_relativeViewFusion_workers8' 2>/dev/null || true
pkill -f 'H129_H76_VFT_gifformerMask05' 2>/dev/null || true
echo "[lit-dedup] stopped H122 (subset of H130), H129 (dup H43-H45 mask on H35)"

# Cancelled: h129, h133, h134, h131 until H113/H127 evaluated
# H130: do not start if already training
if ! pgrep -af train_rumpl | grep -q 'H130_H76_relViewFusion_gifformerMask05'; then
  run_one 1 h130_relview_mask05 H130_H76_relViewFusion_gifformerMask05_w322_workers6_seed0_20260805 || true
fi

# H132 only if not duplicate of running H123 (JV2 vs JV1+bias — different)
run_one 0 h132_gjv1_biased H132_H76_globalJV1_gbtBiased_rezero_w322_workers6_seed0_20260805 || true

echo "[lit-dedup] deferred H131 until H113+H127 table2; see ${REG}"
date --iso-8601=seconds >"${ROOT}/queue_literature_fusion_20260805.done"
echo "[lit-dedup] finished $(date --iso-8601=seconds)"
