#!/usr/bin/env bash
# Master post-nv4 schedule: H147/H148 -> H141-H146 -> H149-H152 -> H156/H161 nv4 -> H165-H168 -> H153 reeval.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/queue_master_nv4_20260806.log
LOCK=${ROOT}/queue_master_nv4.lock
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
ARCH_BASE=${ROOT}/H141_H152_arch_sprint

mkdir -p "${ROOT}" "${ARCH_BASE}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "[master] another master queue holds ${LOCK}; exit"
  exit 0
fi
exec >>"${LOG}" 2>&1

export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_EVAL_STRICT=0

echo "======== master nv4 queue $(date --iso-8601=seconds) ========"

stop_contention() {
  echo "[master] stop stale arch queue processes $(date --iso-8601=seconds)"
  pkill -f 'queue_H141_H152_arch_sprint_parallel_20260805.sh' 2>/dev/null || true
  pkill -f 'restart_H147_H148_nv4_20260806.sh' 2>/dev/null || true
  sleep 2
}

wait_tag() {
  local tag=$1
  local done_file=${ARCH_BASE}/completed/${tag}.done
  local ckpt_file=${ARCH_BASE}/checkpoints/${tag}.txt
  while true; do
    if [[ -s "${done_file}" ]]; then
      echo "[master] done ${tag} $(date --iso-8601=seconds)"
      return 0
    fi
    if ! pgrep -f "exp-name ${tag}" >/dev/null 2>&1; then
      if [[ -s "${ckpt_file}" ]] && [[ -s "$(tr -d '\r\n' <"${ckpt_file}")" ]]; then
        echo "[master] ${tag} train idle with ckpt; waiting for .done or launch eval-only"
      fi
      # launch script may still be in eval phase
      if pgrep -f "launch_H59.*${tag}" >/dev/null 2>&1; then
        sleep 60
        continue
      fi
      if [[ -s "${done_file}" ]]; then
        return 0
      fi
      echo "[master] WARN ${tag} not running and not done"
      return 1
    fi
    sleep 120
  done
}

run_arch_pair() {
  local a=$1 b=$2
  bash "${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh" "${a}" 0 \
    || echo "[master] WARN ${a} gpu0 failed" &
  local pa=$!
  bash "${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh" "${b}" 1 \
    || echo "[master] WARN ${b} gpu1 failed" &
  local pb=$!
  wait "${pa}" || true
  wait "${pb}" || true
}

run_arch_one() {
  bash "${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh" "$1" 0 \
    || echo "[master] WARN $1 failed"
}

clear_partial() {
  local prefix=$1
  find "${MODEL_OUTPUT}" -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null \
    | while read -r d; do rm -rf "${d}" 2>/dev/null || true; done
  rm -f "${ARCH_BASE}/checkpoints/${prefix}.txt" "${ARCH_BASE}/completed/${prefix}.done"
  rm -f "${ROOT}/H153_H164_radical_sprint/checkpoints/${prefix}.txt" \
    "${ROOT}/H153_H164_radical_sprint/completed/${prefix}.done"
}

run_radical_nv4() {
  local variant=$1 gpu=$2 new_tag=$3
  export RUMPL_TAG_OVERRIDE="${new_tag}"
  bash "${AUDIT}/launch_H153_H164_radical_sprint_20260805.sh" "${variant}" "${gpu}" \
    || echo "[master] WARN ${new_tag} failed"
  unset RUMPL_TAG_OVERRIDE
}

# --- Phase 0: trim contention (keep running H147/H148 if present) ---
stop_contention
for prefix in \
  H149_H81_gateGlobalJV2_w322_ftH81_workers8_seed0_20260805 \
  H150_H81_gateRelView_w322_ftH81_workers8_seed0_20260805
do
  if [[ ! -s "${ARCH_BASE}/completed/${prefix}.done" ]]; then
    if pgrep -f "exp-name ${prefix}" >/dev/null 2>&1; then
      echo "[master] defer ${prefix} (stop until phase3)"
      pkill -f "exp-name ${prefix}" 2>/dev/null || true
    fi
    clear_partial "${prefix}"
  fi
done
sleep 2

# --- Phase 1: H147 + H148 (skip if already .done) ---
for pair in "h147_vft_mask04 0 H147_H81_vftMask04_w322_ftH81_workers8_seed0_20260805" \
            "h148_jv1_biased_h81 1 H148_H81_globalJV1_confgeom_w322_ftH81_workers8_seed0_20260805"; do
  read -r var gpu tag <<<"${pair}"
  if [[ -s "${ARCH_BASE}/completed/${tag}.done" ]]; then
    echo "[master] skip phase1 ${tag} (done)"
    continue
  fi
  if ! pgrep -f "exp-name ${tag}" >/dev/null 2>&1; then
    echo "[master] launch ${tag} gpu${gpu}"
    bash "${AUDIT}/launch_H141_H152_arch_sprint_20260805.sh" "${var}" "${gpu}" &
  fi
done
wait
for tag in \
  H147_H81_vftMask04_w322_ftH81_workers8_seed0_20260805 \
  H148_H81_globalJV1_confgeom_w322_ftH81_workers8_seed0_20260805
do
  wait_tag "${tag}" || true
done

# --- Phase 2: H141-H146 retry ---
echo "[master] phase2 H141-H146 $(date --iso-8601=seconds)"
bash "${AUDIT}/queue_H141_H146_retry_nv4_20260806.sh" || true

# --- Phase 3: H149–H152 (kill partial if still epoch<8 from pre-fix contention) ---
for prefix in \
  H149_H81_gateGlobalJV2_w322_ftH81_workers8_seed0_20260805 \
  H150_H81_gateRelView_w322_ftH81_workers8_seed0_20260805
do
  if [[ ! -s "${ARCH_BASE}/completed/${prefix}.done" ]]; then
    if pgrep -f "exp-name ${prefix}" >/dev/null 2>&1; then
      echo "[master] stop contested ${prefix} for clean nv4 rerun"
      pkill -f "exp-name ${prefix}" 2>/dev/null || true
      sleep 2
    fi
    clear_partial "${prefix}"
  fi
done

echo "[master] phase3 H149-H152 $(date --iso-8601=seconds)"
run_arch_pair h149_gate_gjv2_w322 h150_gate_relview_w322
for tag in \
  H149_H81_gateGlobalJV2_w322_ftH81_workers8_seed0_20260805 \
  H150_H81_gateRelView_w322_ftH81_workers8_seed0_20260805
do wait_tag "${tag}" || true; done

run_arch_pair h151_bone_ray01 h152_shallow_vft
for tag in \
  H151_H81_boneRay01_w322_ftH81_workers8_seed0_20260805 \
  H152_H81_singlePftBlock_w322_ftH81_workers8_seed0_20260805
do wait_tag "${tag}" || true; done

# --- Phase 4: H156 + H161 full nv4 retrain ---
echo "[master] phase4 H156/H161 nv4 $(date --iso-8601=seconds)"
H156_TAG=H156_H81_vftDepth1_w322_nv4_workers12_seed0_20260806
H161_TAG=H161_H81_vft1_skipPft_w322_nv4_workers12_seed0_20260806
clear_partial "${H156_TAG}"
clear_partial "H156_H81_vftDepth1_w322_ftH81_workers12_seed0_20260805"
clear_partial "${H161_TAG}"
clear_partial "H161_H81_vft1_skipPft_w322_ftH81_workers12_seed0_20260805"
run_radical_nv4 h156_vft1 0 "${H156_TAG}" &
run_radical_nv4 h161_vft1_skip_pft 1 "${H161_TAG}" &
wait

# --- Phase 5: H165-H168 ---
echo "[master] phase5 H165-H168 $(date --iso-8601=seconds)"
bash "${AUDIT}/queue_H165_H168_nv4_20260806.sh" || true

# --- Phase 6: reeval H153-H164 V3/V4 (adapt load; legacy train still 2-view) ---
echo "[master] phase6 reeval H153-H164 V3/V4 $(date --iso-8601=seconds)"
bash "${AUDIT}/reeval_H153_H164_V234_20260806.sh" 0 || true

python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
targets = {2: 36.8, 3: 30.4, 4: 26.0}
print("\n=== master queue summary (All-17 mm) ===")
for base_name in (
    "H141_H152_arch_sprint/eval",
    "H153_H164_radical_sprint/eval",
    "H165_H168_nv4_w322/eval",
):
    base = root / base_name
    if not base.is_dir():
        continue
    for sub in sorted(base.glob("H*")):
        vals = {}
        for v in (2, 3, 4):
            f = sub / f"V{v}/table2.json"
            if f.is_file():
                vals[v] = json.load(open(f))["table2_action_equal"]["all17_mm"]
        if not vals:
            continue
        beat = sum(1 for k, t in targets.items() if k in vals and vals[k] < t)
        s = " ".join(f"V{k}={vals[k]:.2f}" for k in sorted(vals))
        print(f"  {sub.name[:36]:36s} {s}  beat={beat}/3")
PY

echo "[master] finished $(date --iso-8601=seconds)"
