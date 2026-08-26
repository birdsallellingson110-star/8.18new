#!/usr/bin/env bash
set -euo pipefail
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260819
LOG=${ROOT}/h21_wait_launch.log
LAUNCH=${AUDIT}/launch_h21_occlusion_ray_temporal_20260819.sh
FREE_MIB_NEEDED=16000
POLL_SECONDS=60
mkdir -p "${ROOT}"
variants=(h21a_missing_dropout h21b_missing_mixste_loss)
declare -A out_name=(
  [h21a_missing_dropout]=h21a_prevft_missing_dropout
  [h21b_missing_mixste_loss]=h21b_prevft_missing_mixste_loss
)
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG}"; }
gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}
is_done() { [[ -s "${ROOT}/${out_name[$1]}/COMPLETED" ]]; }
is_running() {
  pgrep -af "launch_h21_occlusion_ray_temporal_20260819.sh ${1} " >/dev/null 2>&1
}
busy_gpus() {
  pgrep -af "launch_h21_occlusion_ray_temporal_20260819.sh" | awk '{
    for (i=1;i<=NF;i++) if ($i ~ /^[01]$/) print $i
  }' | sort -u
}
log "H21 waiter started; variants=${variants[*]}"
while true; do
  pending=()
  for variant in "${variants[@]}"; do
    is_done "${variant}" && continue
    is_running "${variant}" && continue
    pending+=("${variant}")
  done
  if [[ ${#pending[@]} -eq 0 ]]; then
    still=0
    for variant in "${variants[@]}"; do
      is_running "${variant}" && still=1
    done
    if [[ ${still} -eq 0 ]]; then
      log "all H21 completed"
      exit 0
    fi
    sleep "${POLL_SECONDS}"
    continue
  fi
  occupied=$(busy_gpus || true)
  for gpu in 0 1; do
    [[ ${#pending[@]} -eq 0 ]] && break
    echo "${occupied}" | grep -qx "${gpu}" && continue
    free=$(gpu_free_mib "${gpu}")
    [[ "${free}" -lt "${FREE_MIB_NEEDED}" ]] && continue
    variant=${pending[0]}
    pending=("${pending[@]:1}")
    mkdir -p "${ROOT}/${out_name[$variant]}"
    log "launch ${variant} on GPU${gpu} (free=${free} MiB)"
    nohup bash "${LAUNCH}" "${variant}" "${gpu}" \
      >>"${ROOT}/${out_name[$variant]}/waiter.out" 2>&1 &
    occupied="${occupied}"$'\n'"${gpu}"
    sleep 8
  done
  sleep "${POLL_SECONDS}"
done
