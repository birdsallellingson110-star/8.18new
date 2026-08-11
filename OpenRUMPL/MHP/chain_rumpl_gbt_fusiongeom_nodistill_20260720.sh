#!/usr/bin/env bash
set -euo pipefail

TRAIN=/home/lixiaob/cjy/OpenRUMPL/MHP/run_rumpl_gbt_fusiongeom_nodistill_20260720.sh
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh
SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_combo_eval.py
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
REPORT=/mnt/data/cjyoutput/rumpl_gbt_fusiongeom_nodistill_eval_summary_20260720.txt
CHAIN_LOG=/mnt/data/cjyoutput/rumpl_gbt_fusiongeom_nodistill_chain_20260720.log
GPU=0
V2_RUMPL='3_6=40.37,3_12=46.95,3_13=39.79,3_23=32.30,6_12=67.28,6_13=53.52,6_23=39.39,12_13=59.41,12_23=46.04,13_23=44.08'
RUNS=(geom:g1:1.0 geom:g01:0.1 both:g1:1.0 both:g01:0.1)

exec > >(tee -a "$CHAIN_LOG") 2>&1
echo "[train] $(date '+%F %T') direct [FUS]-to-view geometry ablations, no distillation"
pids=()
for spec in "${RUNS[@]}"; do
  IFS=: read -r mode tag init <<< "$spec"
  "$TRAIN" "$mode" "$init" "$GPU" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "[train-done] $(date '+%F %T')"

find_run() {
  local mode=$1 tag=$2
  find "$ROOT" -maxdepth 1 -type d \
    -name "rumpl_gbt_fusiongeom_nodistill_${mode}_${tag}_20260720_*" -print | sort | tail -n 1
}

evaluate_run() {
  local mode=$1 tag=$2 init=$3
  local use_conf=0 run checkpoint name
  [[ $mode == both ]] && use_conf=1
  run=$(find_run "$mode" "$tag")
  checkpoint="$run/final_state.pth.tar"
  name="rumpl_gbt_fusiongeom_nodistill_${mode}_${tag}_final20"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing final checkpoint: $checkpoint" >&2
    exit 1
  fi

  export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=$use_conf GBT_USE_GEOM_BIAS=1
  export GBT_CONF_INIT=0.1 GBT_GEOM_INIT="$init" GBT_FUSION_GEOM=1
  export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0
  unset GBT_V2_SCALE GBT_V3_SCALE GBT_V4_SCALE
  "$EVAL" "$checkpoint" "$name" 2 "/mnt/data/cjyoutput/cmu_v2_eval_${name}_20260720" "$GPU"
  "$EVAL" "$checkpoint" "$name" 3 "/mnt/data/cjyoutput/cmu_v3_eval_${name}_20260720" "$GPU"
  "$EVAL" "$checkpoint" "$name" 4 "/mnt/data/cjyoutput/cmu_v4_eval_${name}_20260720" "$GPU"
}

for spec in "${RUNS[@]}"; do
  IFS=: read -r mode tag init <<< "$spec"
  evaluate_run "$mode" "$tag" "$init"
done

: > "$REPORT"
for spec in "${RUNS[@]}"; do
  IFS=: read -r mode tag init <<< "$spec"
  name="rumpl_gbt_fusiongeom_nodistill_${mode}_${tag}_final20"
  for k in 2 3 4; do
    echo "=== ${name} V${k} ===" >> "$REPORT"
    if [[ $k == 2 ]]; then
      "$PY" "$SUMMARY" "/mnt/data/cjyoutput/cmu_v2_eval_${name}_20260720" \
        --rumpl-values "$V2_RUMPL" \
        --baseline /mnt/data/cjyoutput/cmu_v2_eval_hardv_legw09_full_20260712_fg \
        --name "$name" >> "$REPORT"
    elif [[ $k == 3 ]]; then
      "$PY" "$SUMMARY" "/mnt/data/cjyoutput/cmu_v3_eval_${name}_20260720" \
        --rumpl /mnt/data/cjyoutput/cmu_v3_eval_org_20260711 \
        --baseline /mnt/data/cjyoutput/cmu_v3_eval_hardv_legw09_full_20260714 \
        --name "$name" >> "$REPORT"
    else
      "$PY" "$SUMMARY" "/mnt/data/cjyoutput/cmu_v4_eval_${name}_20260720" \
        --rumpl /mnt/data/cjyoutput/cmu_v4_eval_rumpl_conf_20260714 \
        --baseline /mnt/data/cjyoutput/cmu_v4_eval_hardv_legw09_full_20260714 \
        --name "$name" >> "$REPORT"
    fi
  done
done
echo "[finish] $(date '+%F %T') report=$REPORT"
