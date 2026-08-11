#!/usr/bin/env bash
# Opportunistic strict H36M generator worker.
# Runs high-numbered splits in descending order to avoid colliding with the
# main low-to-high scheduler that is already running.
set -euo pipefail

START=${1:-98}
STOP=${2:-87}
SUBSET=${3:-train}
GPU=${4:-1}
MIN_SAMPLES=${MIN_SAMPLES:-1000}

LOGDIR=/mnt/data/cjydata/step2_logs_h36m_${SUBSET}_gpu${GPU}_tail_${START}_${STOP}
SV=/mnt/data/cjydata/mhp_workspace/paper_single_h36m/stage_V/${SUBSET}

mkdir -p "$LOGDIR" "$SV"
echo "=== H36M ${SUBSET} tail worker start: ${START}..${STOP} on GPU ${GPU} $(date '+%F %T') ===" \
  >> "$LOGDIR/worker.log"

for ((s=START; s>=STOP; s--)); do
  if find "$SV" -maxdepth 1 -type f -name "split_${s}_*.pkl" | grep -q .; then
    echo "[tail][gpu ${GPU}] split ${s} final exists, skip $(date '+%F %T')" >> "$LOGDIR/worker.log"
    continue
  fi
  if find "$SV" -maxdepth 1 -type d -name "split_${s}_*temp_files" | grep -q .; then
    echo "[tail][gpu ${GPU}] split ${s} has temp dir, skip to avoid possible overlap $(date '+%F %T')" \
      >> "$LOGDIR/worker.log"
    continue
  fi

  echo "[tail][gpu ${GPU}] >>> split ${s} start $(date '+%F %T')" >> "$LOGDIR/worker.log"
  bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step2_split_h36m.sh "$s" "$GPU" "$SUBSET" \
    >> "$LOGDIR/split_${s}.log" 2>&1

  out_file=$(find "$SV" -maxdepth 1 -type f -name "split_${s}_*.pkl" | head -n 1)
  if [[ -z "$out_file" ]]; then
    echo "[tail][gpu ${GPU}] !!! split ${s} produced no final pkl $(date '+%F %T')" \
      | tee -a "$LOGDIR/split_${s}.log" "$LOGDIR/worker.log"
    exit 2
  fi
  echo "[tail][gpu ${GPU}] validating ${out_file} $(date '+%F %T')" \
    | tee -a "$LOGDIR/split_${s}.log" "$LOGDIR/worker.log"
  /home/lixiaob/cjy/rumpl_venv310/bin/python /home/lixiaob/cjy/OpenRUMPL/MHP/08_validate_h36m_stagev.py \
    --min-samples "$MIN_SAMPLES" "$out_file" >> "$LOGDIR/split_${s}.log" 2>&1
  echo "[tail][gpu ${GPU}] <<< split ${s} done $(date '+%F %T')" >> "$LOGDIR/worker.log"
done

echo "=== H36M ${SUBSET} tail worker finished $(date '+%F %T') ===" >> "$LOGDIR/worker.log"
