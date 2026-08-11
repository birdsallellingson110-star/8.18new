#!/usr/bin/env bash
set -euo pipefail

TRAIN=/home/lixiaob/cjy/OpenRUMPL/MHP/run_rumpl_gbt_global_jv_gated_20260722.sh
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh
SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_combo_eval.py
PARAM_SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_checkpoint_parameter.py
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
REPORT=/mnt/data/cjyoutput/rumpl_gbt_global_jv_gated_ablation_20260722.txt
LOG=/mnt/data/cjyoutput/rumpl_gbt_global_jv_gated_ablation_20260722.log
V2_RUMPL='3_6=40.37,3_12=46.95,3_13=39.79,3_23=32.30,6_12=67.28,6_13=53.52,6_23=39.39,12_13=59.41,12_23=46.04,13_23=44.08'

exec > >(tee -a "$LOG") 2>&1
echo "[start] $(date '+%F %T') gated global joint-view ablation"
"$TRAIN" nodrop 0 & nodrop_pid=$!
"$TRAIN" stage5drop 1 & stage5drop_pid=$!
wait "$nodrop_pid"
wait "$stage5drop_pid"
echo "[train-done] $(date '+%F %T')"

find_run() {
  local mode=$1
  find "$ROOT" -maxdepth 1 -type d \
    -name "rumpl_gbt_global_jv_gated_${mode}_20260722_*" -print | sort | tail -n1
}

evaluate_mode() (
  set -euo pipefail
  local mode=$1 gpu=$2 run checkpoint name k
  run=$(find_run "$mode")
  checkpoint="$run/final_state.pth.tar"
  name="rumpl_gbt_global_jv_gated_${mode}_final20"
  [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 1; }
  export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
  export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=1.0 GBT_FUSION_GEOM=0
  export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0
  export GBT_GLOBAL_JV_DEPTH=1 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=1
  export GBT_TOKEN_DROPOUT=0
  for k in 2 3 4 5; do
    "$EVAL" "$checkpoint" "$name" "$k" \
      "/mnt/data/cjyoutput/cmu_v${k}_eval_${name}_20260722" "$gpu"
  done
)

evaluate_mode nodrop 0 & nodrop_eval_pid=$!
evaluate_mode stage5drop 1 & stage5drop_eval_pid=$!
wait "$nodrop_eval_pid"
wait "$stage5drop_eval_pid"

: > "$REPORT"
for mode in nodrop stage5drop; do
  name="rumpl_gbt_global_jv_gated_${mode}_final20"
  for k in 2 3 4 5; do
    candidate="/mnt/data/cjyoutput/cmu_v${k}_eval_${name}_20260722"
    if [[ "$mode" == nodrop ]]; then
      baseline="/mnt/data/cjyoutput/cmu_v${k}_eval_rumpl_gbt_curriculum5_nodistill_both_final20_20260720"
    else
      baseline="/mnt/data/cjyoutput/cmu_v${k}_eval_rumpl_gbt_tokendrop0p1_stage5_seed0_final20_20260721"
    fi
    echo "=== ${name} V${k} ===" >> "$REPORT"
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
echo "=== learned global gates ===" >> "$REPORT"
for mode in nodrop stage5drop; do
  run=$(find_run "$mode")
  echo "--- $mode ---" >> "$REPORT"
  "$PY" "$PARAM_SUMMARY" "$run/final_state.pth.tar" global_jv_gate >> "$REPORT"
done
echo "[finish] $(date '+%F %T') report=$REPORT"
