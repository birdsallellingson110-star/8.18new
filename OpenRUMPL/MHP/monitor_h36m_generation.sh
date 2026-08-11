#!/usr/bin/env bash
# Quick status view for strict H36M generation.
set -euo pipefail
SUBSET=${1:-train}
GPU=${2:-1}
BASE=/mnt/data/cjydata/mhp_workspace/paper_single_h36m/stage_V/${SUBSET}
LOGDIR=/mnt/data/cjydata/step2_logs_h36m_${SUBSET}_gpu${GPU}

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "=== processes ==="
pgrep -af 'run_step2_h36m|run_step2_split_h36m|run_mmpose_02_run.py|train_rumpl.py' || true
echo
echo "=== files ==="
echo -n "final pkl: "
find "$BASE" -maxdepth 1 -type f -name '*.pkl' 2>/dev/null | wc -l
echo -n "temp pkl:  "
find "$BASE" -maxdepth 2 -type f -path '*temp_files/temp_*.pkl' 2>/dev/null | wc -l
echo
echo "=== latest worker tails ==="
for f in "$LOGDIR"/worker_*.log; do
  [ -e "$f" ] || continue
  echo "--- $f"
  tail -6 "$f"
done
echo
echo "=== latest split progress ==="
for f in "$LOGDIR"/split_*.log; do
  [ -e "$f" ] || continue
  echo "--- $f"
  tail -3 "$f"
done | tail -80
