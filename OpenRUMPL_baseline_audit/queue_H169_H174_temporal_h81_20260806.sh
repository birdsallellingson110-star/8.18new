#!/usr/bin/env bash
# H169-H174 temporal queue: wait for GPU, identity check, then 2-wide waves.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
LAUNCH="${AUDIT}/launch_H169_H174_temporal_h81_gbt_aligned_20260806.sh"
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
LOG=${ROOT}/H169_H174_temporal_h81_gbt_aligned/queue.log
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
H81=$(tr -d '\r\n' < "${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt")

mkdir -p "${ROOT}/H169_H174_temporal_h81_gbt_aligned/logs"
exec >>"${LOG}" 2>&1
echo "======== temporal H81 queue $(date --iso-8601=seconds) ========"
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4

wait_gpu() {
  local gpu=$1
  while true; do
    if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      echo "[temporal] wait GPU${gpu} $(date --iso-8601=seconds)"
      sleep 180
      continue
    fi
    break
  done
}

for mode in mixste-ttb mixste-ttb-residual mixste-alternating; do
  echo "[temporal] identity ${mode} $(date --iso-8601=seconds)"
  if ! "${PY}" -u "${REPO}/run/check_temporal_h76_identity.py" \
    --cfg "${CFG}" --checkpoint "${H81}" --backbone-flavor h81 \
    --fusion-mode "${mode}" --depth 3 --time 9 --views 2 --device cuda:0; then
    echo "[temporal] WARN identity ${mode} (H81 path offset; monitor step-100 train_mpjpe in log)"
  fi
done

VARIANTS=(
  h169_mixste_ttb_frozen
  h170_mixste_ttb_res
  h171_mixste_alt
  h172_mixste_unfreeze_vft
  h174_mixste_ttb_v4train
  h173_global_mixste_loss
)

run_one() {
  local gpu=$1 var=$2
  wait_gpu "${gpu}"
  echo "[temporal] GPU${gpu} ${var} start $(date --iso-8601=seconds)"
  bash "${LAUNCH}" "${var}" "${gpu}" \
    || echo "[temporal] WARN ${var} failed"
  echo "[temporal] GPU${gpu} ${var} done $(date --iso-8601=seconds)"
}

i=0
while [[ $i -lt ${#VARIANTS[@]} ]]; do
  a=${VARIANTS[$i]}
  b=${VARIANTS[$((i + 1))]:-}
  if [[ -z "${b}" ]]; then
    run_one 1 "${a}"
  else
    run_one 0 "${a}" &
    run_one 1 "${b}" &
    wait
  fi
  i=$((i + 2))
done

echo "[temporal] finished $(date --iso-8601=seconds)"
