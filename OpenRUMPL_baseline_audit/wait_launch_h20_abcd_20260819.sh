#!/usr/bin/env bash
# Launch remaining H20 variants when a GPU has enough free memory.
# Do not preempt other users' jobs.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818
LOG=${ROOT}/h20_wait_launch.log
LAUNCH=${AUDIT}/launch_h20_causal_candidate_variant_20260819.sh
FREE_MIB_NEEDED=16000
POLL_SECONDS=60

mkdir -p "${ROOT}"
variants=(h20a_mixste h20b_geom_gate h20c_tloss h20d_joint)
declare -A out_name=(
  [h20a_mixste]=h20a_causal_candidate_mixste
  [h20b_geom_gate]=h20b_causal_candidate_geom_gate
  [h20c_tloss]=h20c_causal_candidate_tloss
  [h20d_joint]=h20d_causal_candidate_joint
)

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG}"; }

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}

is_done() {
  local variant=$1
  [[ -s "${ROOT}/${out_name[$variant]}/COMPLETED" ]]
}

is_running() {
  pgrep -af "launch_h20_causal_candidate_variant_20260819.sh ${1} " >/dev/null 2>&1
}

busy_gpus() {
  pgrep -af "launch_h20_causal_candidate_variant_20260819.sh" | awk '{
    for (i=1;i<=NF;i++) if ($i ~ /^[01]$/) print $i
  }' | sort -u
}

log "H20 waiter started; need ${FREE_MIB_NEEDED} MiB free; variants=${variants[*]}"

while true; do
  pending=()
  for variant in "${variants[@]}"; do
    if is_done "${variant}"; then
      continue
    fi
    if is_running "${variant}"; then
      continue
    fi
    pending+=("${variant}")
  done
  if [[ ${#pending[@]} -eq 0 ]]; then
    still_running=0
    for variant in "${variants[@]}"; do
      if is_running "${variant}"; then
        still_running=1
      fi
    done
    if [[ ${still_running} -eq 0 ]]; then
      log "all H20A/B/C/D completed"
      exit 0
    fi
    sleep "${POLL_SECONDS}"
    continue
  fi

  occupied=$(busy_gpus || true)
  for gpu in 0 1; do
    if [[ ${#pending[@]} -eq 0 ]]; then
      break
    fi
    if echo "${occupied}" | grep -qx "${gpu}"; then
      continue
    fi
    free=$(gpu_free_mib "${gpu}")
    if [[ "${free}" -lt "${FREE_MIB_NEEDED}" ]]; then
      continue
    fi
    variant=${pending[0]}
    pending=("${pending[@]:1}")
    out=${ROOT}/${out_name[$variant]}
    mkdir -p "${out}"
    : > "${out}/train.log"
    log "launch ${variant} on GPU${gpu} (free=${free} MiB)"
    nohup bash "${LAUNCH}" "${variant}" "${gpu}" >>"${out}/waiter.out" 2>&1 &
    occupied="${occupied}"$'\n'"${gpu}"
    sleep 8
  done
  sleep "${POLL_SECONDS}"
done
