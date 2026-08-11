#!/usr/bin/env bash
# Validate completed strict H36M stage_V split files as they appear.
set -euo pipefail

SUBSET=${1:-train}
EXPECTED=${2:-99}
SLEEP_SEC=${SLEEP_SEC:-300}
MIN_SAMPLES=${MIN_SAMPLES:-1000}

SV=/mnt/data/cjydata/mhp_workspace/paper_single_h36m/stage_V/${SUBSET}
LOGDIR=/mnt/data/cjydata/step2_logs_h36m_${SUBSET}_watchdog
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
VALIDATOR=/home/lixiaob/cjy/OpenRUMPL/MHP/08_validate_h36m_stagev.py

mkdir -p "$LOGDIR"
echo "=== H36M ${SUBSET} validator watchdog start $(date '+%F %T') ===" >> "$LOGDIR/watchdog.log"

while true; do
  count=$(find "$SV" -maxdepth 1 -type f -name 'split_*.pkl' | wc -l)
  echo "[watchdog] $(date '+%F %T') final_count=${count}/${EXPECTED}" >> "$LOGDIR/watchdog.log"

  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    base=$(basename "$f")
    ok="$LOGDIR/${base}.validated"
    fail="$LOGDIR/${base}.failed"
    if [[ -e "$ok" || -e "$fail" ]]; then
      continue
    fi
    echo "[watchdog] validating $f $(date '+%F %T')" >> "$LOGDIR/watchdog.log"
    if "$PY" "$VALIDATOR" --min-samples "$MIN_SAMPLES" "$f" >> "$LOGDIR/watchdog.log" 2>&1; then
      date '+%F %T' > "$ok"
    else
      date '+%F %T' > "$fail"
      echo "[watchdog] FAILED $f; see $LOGDIR/watchdog.log" >> "$LOGDIR/watchdog.log"
    fi
  done < <(find "$SV" -maxdepth 1 -type f -name 'split_*.pkl' | sort)

  failed=$(find "$LOGDIR" -maxdepth 1 -type f -name '*.failed' | wc -l)
  if [[ "$failed" -gt 0 ]]; then
    echo "[watchdog] stopping because ${failed} validation failure(s) exist $(date '+%F %T')" >> "$LOGDIR/watchdog.log"
    exit 3
  fi
  if [[ "$count" -ge "$EXPECTED" ]]; then
    echo "[watchdog] all expected split files observed $(date '+%F %T')" >> "$LOGDIR/watchdog.log"
    exit 0
  fi
  sleep "$SLEEP_SEC"
done
