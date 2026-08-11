#!/usr/bin/env bash
# Resume master queue from phase2 (H147/H148 already done).
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_master_nv4_resume_20260806.log
LOCK=${ROOT}/queue_master_nv4.lock
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
ARCH_BASE=${ROOT}/H141_H152_arch_sprint

mkdir -p "${ROOT}"
exec 9>"${LOCK}"
if ! flock -n 9; then echo "[resume] lock busy"; exit 0; fi
exec >>"${LOG}" 2>&1

export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_EVAL_STRICT=0

echo "======== resume from phase2 $(date --iso-8601=seconds) ========"

clear_partial() {
  local prefix=$1
  find "${MODEL_OUTPUT}" -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null \
    | while read -r d; do rm -rf "${d}" 2>/dev/null || true; done
  rm -f "${ARCH_BASE}/checkpoints/${prefix}.txt" "${ARCH_BASE}/completed/${prefix}.done"
  rm -f "${ROOT}/H153_H164_radical_sprint/checkpoints/${prefix}.txt" \
    "${ROOT}/H153_H164_radical_sprint/completed/${prefix}.done"
}

wait_tag() {
  local tag=$1
  local done_file=${ARCH_BASE}/completed/${tag}.done
  while [[ ! -s "${done_file}" ]]; do
    if pgrep -f "exp-name ${tag}" >/dev/null 2>&1; then
      sleep 120
      continue
    fi
    sleep 30
    [[ -s "${done_file}" ]] && break
    echo "[resume] WARN ${tag} stalled"
    return 1
  done
  echo "[resume] done ${tag}"
}

run_arch_pair() {
  local a=$1 b=$2
  bash "${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh" "${a}" 0 || true &
  local pa=$!
  bash "${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh" "${b}" 1 || true &
  local pb=$!
  wait "${pa}" || true
  wait "${pb}" || true
}

run_radical_nv4() {
  local variant=$1 gpu=$2 new_tag=$3
  export RUMPL_TAG_OVERRIDE="${new_tag}"
  bash "${AUDIT}/launch_H153_H164_radical_sprint_20260805.sh" "${variant}" "${gpu}" || true
  unset RUMPL_TAG_OVERRIDE
}

bash "${AUDIT}/queue_H141_H146_retry_nv4_20260806.sh" || true

for prefix in \
  H149_H81_gateGlobalJV2_w322_ftH81_workers8_seed0_20260805 \
  H150_H81_gateRelView_w322_ftH81_workers8_seed0_20260805
do
  clear_partial "${prefix}"
done
run_arch_pair h149_gate_gjv2_w322 h150_gate_relview_w322
for tag in H149_H81_gateGlobalJV2_w322_ftH81_workers8_seed0_20260805 \
  H150_H81_gateRelView_w322_ftH81_workers8_seed0_20260805; do wait_tag "${tag}" || true; done

run_arch_pair h151_bone_ray01 h152_shallow_vft
for tag in H151_H81_boneRay01_w322_ftH81_workers8_seed0_20260805 \
  H152_H81_singlePftBlock_w322_ftH81_workers8_seed0_20260805; do wait_tag "${tag}" || true; done

H156_TAG=H156_H81_vftDepth1_w322_nv4_workers12_seed0_20260806
H161_TAG=H161_H81_vft1_skipPft_w322_nv4_workers12_seed0_20260806
clear_partial "${H156_TAG}"
clear_partial "H156_H81_vftDepth1_w322_ftH81_workers12_seed0_20260805"
clear_partial "${H161_TAG}"
clear_partial "H161_H81_vft1_skipPft_w322_ftH81_workers12_seed0_20260805"
run_radical_nv4 h156_vft1 0 "${H156_TAG}" &
run_radical_nv4 h161_vft1_skip_pft 1 "${H161_TAG}" &
wait

bash "${AUDIT}/queue_H165_H168_nv4_20260806.sh" || true
bash "${AUDIT}/reeval_H153_H164_V234_20260806.sh" 0 || true

echo "[resume] finished $(date --iso-8601=seconds)"
