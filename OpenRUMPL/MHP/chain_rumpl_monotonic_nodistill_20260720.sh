#!/usr/bin/env bash
set -euo pipefail

TRAIN=/home/lixiaob/cjy/OpenRUMPL/MHP/run_rumpl_monotonic_nodistill_20260720.sh
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh
SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_combo_eval.py
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
REPORT=/mnt/data/cjyoutput/rumpl_monotonic_nodistill_eval_summary_20260720.txt
LOG=/mnt/data/cjyoutput/rumpl_monotonic_nodistill_chain_20260720.log
V2_RUMPL='3_6=40.37,3_12=46.95,3_13=39.79,3_23=32.30,6_12=67.28,6_13=53.52,6_23=39.39,12_13=59.41,12_23=46.04,13_23=44.08'
RUNS=(none:0.25:0 both:0.1:0 both:0.25:1 both:0.5:1)

exec > >(tee -a "$LOG") 2>&1
echo "[train] $(date '+%F %T') unified nested-view monotonic experiments"
pids=()
for spec in "${RUNS[@]}"; do
  IFS=: read -r mode weight gpu <<< "$spec"
  "$TRAIN" "$mode" "$weight" "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[train-done] $(date '+%F %T')"

find_run() {
  local mode=$1 weight=$2 tag
  tag=${weight//./p}
  find "$ROOT" -maxdepth 1 -type d \
    -name "rumpl_monotonic_nodistill_${mode}_w${tag}_20260720_*" -print | sort | tail -n1
}

for spec in "${RUNS[@]}"; do
  IFS=: read -r mode weight gpu <<< "$spec"
  tag=${weight//./p}
  run=$(find_run "$mode" "$weight")
  checkpoint="$run/final_state.pth.tar"
  name="rumpl_monotonic_nodistill_${mode}_w${tag}_final20"
  [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  if [[ $mode == both ]]; then
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
  else
    export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0
  fi
  export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=0.1 GBT_FUSION_GEOM=1
  export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0
  for k in 2 3 4 5; do
    "$EVAL" "$checkpoint" "$name" "$k" "/mnt/data/cjyoutput/cmu_v${k}_eval_${name}_20260720" "$gpu"
  done
done

: > "$REPORT"
for spec in "${RUNS[@]}"; do
  IFS=: read -r mode weight gpu <<< "$spec"
  tag=${weight//./p}
  name="rumpl_monotonic_nodistill_${mode}_w${tag}_final20"
  for k in 2 3 4 5; do
    echo "=== ${name} V${k} ===" >> "$REPORT"
    args=("/mnt/data/cjyoutput/cmu_v${k}_eval_${name}_20260720" --name "$name")
    if [[ $k == 2 ]]; then
      args+=(--rumpl-values "$V2_RUMPL")
    elif [[ $k == 3 ]]; then
      args+=(--rumpl /mnt/data/cjyoutput/cmu_v3_eval_org_20260711)
    elif [[ $k == 4 ]]; then
      args+=(--rumpl /mnt/data/cjyoutput/cmu_v4_eval_rumpl_conf_20260714)
    else
      args+=(--rumpl /mnt/data/cjyoutput/cmu_v5_eval_rumpl_conf_20260720)
    fi
    "$PY" "$SUMMARY" "${args[@]}" >> "$REPORT"
  done
done
echo "[finish] $(date '+%F %T') report=$REPORT"
