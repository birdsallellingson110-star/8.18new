#!/usr/bin/env bash
# Wait for Stage 2a training to finish, then run the occlusion eval sweep and print
# the S2a-vs-R5 comparison. Safe to run in background.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
train_pid=${1:?usage: $0 TRAIN_PID}
variant=${2:-S2a_structocc_L04_seed0_20260724}

echo "[chain] waiting for training pid $train_pid to finish ..."
while kill -0 "$train_pid" 2>/dev/null; do sleep 60; done
echo "[chain] training pid $train_pid ended at $(date --iso-8601=seconds); starting eval"
sleep 10

bash "$repo/run_stage2a_eval_occlusion_20260724.sh" 1 "$variant"

echo; echo "############ S2a vs R5 occlusion comparison (All-17 / KP* mm) ############"
for v in 2 3 4 5; do for occ in 0.0 0.3 0.6; do
  r="$root/R5_v${v}_occ${occ}_summary.json"; s="$root/S2a_v${v}_occ${occ}_summary.json"
  [ -f "$r" ] && [ -f "$s" ] || continue
  ra=$(grep -A2 '"overall"' "$r" | grep all17_mm | grep -oE '[0-9.]+'); rk=$(grep -A3 '"overall"' "$r" | grep kpstar_mm | grep -oE '[0-9.]+')
  sa=$(grep -A2 '"overall"' "$s" | grep all17_mm | grep -oE '[0-9.]+'); sk=$(grep -A3 '"overall"' "$s" | grep kpstar_mm | grep -oE '[0-9.]+')
  printf "V%s occ%s :  R5 %6.2f/%6.2f  ->  S2a %6.2f/%6.2f  (dAll %+.2f  dKP %+.2f)\n" \
    "$v" "$occ" "$ra" "$rk" "$sa" "$sk" \
    "$(echo "$sa - $ra" | bc -l)" "$(echo "$sk - $rk" | bc -l)"
done; done
echo "=== chain_stage2a done $(date --iso-8601=seconds) ==="
