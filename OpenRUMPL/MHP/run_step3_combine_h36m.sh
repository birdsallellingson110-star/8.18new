#!/usr/bin/env bash
# Combine strict H36M stage_V split pkl files.
# Usage: bash run_step3_combine_h36m.sh [subset=train]
set -euo pipefail

SUBSET=${1:-train}
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1
cd /home/lixiaob/cjy/OpenRUMPL/MHP

WORK=/mnt/data/cjydata/mhp_workspace
EXP=paper_single_h36m
SV=$WORK/$EXP/stage_V/$SUBSET

n=$(ls "$SV"/*.pkl 2>/dev/null | grep -v temp | wc -l)
echo "stage_V/$SUBSET final pkl count: $n (need 99)"
if [ "$n" -ne 99 ]; then
  echo "!! $SUBSET is incomplete; wait for generation first."
  exit 1
fi

python -u run_mmpose_04_combine.py \
  --exp "$EXP" \
  --extra-name random_20_small_room_h36m \
  --n-splits 100 \
  --operation-on "$SUBSET" \
  --work-dir "$WORK"

echo
echo "=== $SUBSET combined outputs ==="
find "$WORK/$EXP/datasets" -name "amass_mmpose_joints_${SUBSET}.pkl"
