#!/usr/bin/env bash
set -euo pipefail

SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_combo_eval.py
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPORT=/mnt/data/cjyoutput/rumpl_gbt_curriculum5_both_seed_replicates_20260721.txt
V2_RUMPL='3_6=40.37,3_12=46.95,3_13=39.79,3_23=32.30,6_12=67.28,6_13=53.52,6_23=39.39,12_13=59.41,12_23=46.04,13_23=44.08'

: > "$REPORT"
for seed in 1 2; do
  name="rumpl_gbt_curriculum5_both_seed${seed}_final20"
  for k in 2 3 4 5; do
    candidate="/mnt/data/cjyoutput/cmu_v${k}_eval_${name}_20260721"
    baseline="/mnt/data/cjyoutput/cmu_v${k}_eval_rumpl_gbt_curriculum5_nodistill_both_final20_20260720"
    echo "=== seed=${seed} V${k} ===" >> "$REPORT"
    if [[ $k == 2 ]]; then
      "$PY" "$SUMMARY" "$candidate" --rumpl-values "$V2_RUMPL" --baseline "$baseline" --name "$name" >> "$REPORT"
    elif [[ $k == 3 ]]; then
      "$PY" "$SUMMARY" "$candidate" --rumpl /mnt/data/cjyoutput/cmu_v3_eval_org_20260711 --baseline "$baseline" --name "$name" >> "$REPORT"
    elif [[ $k == 4 ]]; then
      "$PY" "$SUMMARY" "$candidate" --rumpl /mnt/data/cjyoutput/cmu_v4_eval_rumpl_conf_20260714 --baseline "$baseline" --name "$name" >> "$REPORT"
    else
      "$PY" "$SUMMARY" "$candidate" --rumpl /mnt/data/cjyoutput/cmu_v5_eval_rumpl_conf_20260720 --baseline "$baseline" --name "$name" >> "$REPORT"
    fi
  done
done
echo "report=$REPORT"
