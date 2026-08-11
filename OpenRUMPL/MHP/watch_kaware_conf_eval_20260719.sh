#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh
SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_combo_eval.py
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPORT=/mnt/data/cjyoutput/kaware_conf_eval_summary_20260719.txt
V2_RUMPL='3_6=40.37,3_12=46.95,3_13=39.79,3_23=32.30,6_12=67.28,6_13=53.52,6_23=39.39,12_13=59.41,12_23=46.04,13_23=44.08'

find_run() {
  local prefix=$1
  find "$ROOT" -maxdepth 1 -type d -name "${prefix}_*" -print | sort | tail -n 1
}

geom_run=$(find_run distill_dualhard_legw09_gbt_kaware_aux075_20260719)
conf_run=$(find_run distill_dualhard_legw09_gbt_kaware_conf01_aux075_20260719)

while [[ ! -s "$geom_run/final_state.pth.tar" || ! -s "$conf_run/final_state.pth.tar" ]]; do
  sleep 300
done

evaluate_model() {
  local name=$1 checkpoint=$2 conf_bias=$3
  export GBT_CONF_BIAS=$conf_bias GBT_GEOM_BIAS=0.12 GBT_VIEW_AWARE=1
  export GBT_V2_SCALE=0
  "$EVAL" "$checkpoint" "$name" 2 "/mnt/data/cjyoutput/cmu_v2_eval_${name}_20260719" 1
  export GBT_V2_SCALE=1
  "$EVAL" "$checkpoint" "$name" 3 "/mnt/data/cjyoutput/cmu_v3_eval_${name}_20260719" 1
  export GBT_V2_SCALE=2
  "$EVAL" "$checkpoint" "$name" 4 "/mnt/data/cjyoutput/cmu_v4_eval_${name}_20260719" 1
}

evaluate_model kaware_aux075_final20 "$geom_run/final_state.pth.tar" 0
evaluate_model kaware_conf01_aux075_final20 "$conf_run/final_state.pth.tar" 0.1

: > "$REPORT"
for name in kaware_aux075_final20 kaware_conf01_aux075_final20; do
  for k in 2 3 4; do
    echo "=== ${name} V${k} ===" >> "$REPORT"
    if [[ $k == 2 ]]; then
      "$PY" "$SUMMARY" "/mnt/data/cjyoutput/cmu_v2_eval_${name}_20260719" \
        --rumpl-values "$V2_RUMPL" \
        --baseline /mnt/data/cjyoutput/cmu_v2_eval_hardv_legw09_full_20260712_fg \
        --name "$name" >> "$REPORT"
    elif [[ $k == 3 ]]; then
      "$PY" "$SUMMARY" "/mnt/data/cjyoutput/cmu_v3_eval_${name}_20260719" \
        --rumpl /mnt/data/cjyoutput/cmu_v3_eval_org_20260711 \
        --baseline /mnt/data/cjyoutput/cmu_v3_eval_hardv_legw09_full_20260714 \
        --name "$name" >> "$REPORT"
    else
      "$PY" "$SUMMARY" "/mnt/data/cjyoutput/cmu_v4_eval_${name}_20260719" \
        --rumpl /mnt/data/cjyoutput/cmu_v4_eval_rumpl_conf_20260714 \
        --baseline /mnt/data/cjyoutput/cmu_v4_eval_hardv_legw09_full_20260714 \
        --name "$name" >> "$REPORT"
    fi
  done
done
