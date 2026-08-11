#!/usr/bin/env bash
# Conservative strict H36M generator scheduler.
# Uses GPU1 only so GPU0 can keep running model experiments.
set -euo pipefail

NWORKERS=${1:-3}
NSPLITS=${2:-99}
SUBSET=${3:-train}
GPU=${4:-1}
LOGDIR=/mnt/data/cjydata/step2_logs_h36m_${SUBSET}_gpu${GPU}
SV=/mnt/data/cjydata/mhp_workspace/paper_single_h36m/stage_V/${SUBSET}
MIN_SAMPLES=${MIN_SAMPLES:-1000}

mkdir -p "$LOGDIR" "$SV"

run_worker() {
  local w=$1
  for ((s=w; s<NSPLITS; s+=NWORKERS)); do
    if ls "$SV"/split_${s}_*.pkl >/dev/null 2>&1; then
      echo "[h36m ${SUBSET} w${w}] split ${s} exists, skip $(date '+%F %T')"
      continue
    fi
    echo "[h36m ${SUBSET} w${w}][gpu ${GPU}] >>> split ${s} start $(date '+%F %T')"
    bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step2_split_h36m.sh "$s" "$GPU" "$SUBSET" \
      >> "$LOGDIR/split_${s}.log" 2>&1
    out_file=$(find "$SV" -maxdepth 1 -type f -name "split_${s}_*.pkl" ! -path "*temp_files*" | head -n 1)
    if [[ -z "$out_file" ]]; then
      echo "[h36m ${SUBSET} w${w}][gpu ${GPU}] !!! split ${s} produced no final pkl $(date '+%F %T')" \
        | tee -a "$LOGDIR/split_${s}.log"
      exit 2
    fi
    echo "[h36m ${SUBSET} w${w}][gpu ${GPU}] validating ${out_file} $(date '+%F %T')" \
      | tee -a "$LOGDIR/split_${s}.log"
    /home/lixiaob/cjy/rumpl_venv310/bin/python /home/lixiaob/cjy/OpenRUMPL/MHP/08_validate_h36m_stagev.py \
      --min-samples "$MIN_SAMPLES" "$out_file" >> "$LOGDIR/split_${s}.log" 2>&1
    echo "[h36m ${SUBSET} w${w}][gpu ${GPU}] <<< split ${s} done exit=$? $(date '+%F %T')"
  done
}

echo "=== H36M ${SUBSET} scheduler start: ${NWORKERS} workers on GPU ${GPU} $(date '+%F %T') ==="
for ((w=0; w<NWORKERS; w++)); do
  run_worker "$w" >> "$LOGDIR/worker_${w}.log" 2>&1 &
done
wait
echo "=== H36M ${SUBSET} scheduler finished $(date '+%F %T') ==="
