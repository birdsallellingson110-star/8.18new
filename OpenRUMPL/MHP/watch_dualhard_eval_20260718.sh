#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh
SUMMARY=/home/lixiaob/cjy/OpenRUMPL/MHP/summarize_combo_eval.py
REPORT=/mnt/data/cjyoutput/dualhard_eval_summary_20260718.txt
V2_RUMPL='3_6=40.37,3_12=46.95,3_13=39.79,3_23=32.30,6_12=67.28,6_13=53.52,6_23=39.39,12_13=59.41,12_23=46.04,13_23=44.08'

wait_for_model() {
  local exp_name=$1 prefix=$2
  while true; do
    local run_dir
    run_dir=$(find "$ROOT" -maxdepth 1 -type d -name "${prefix}_*" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
    if [[ -n "$run_dir" && -f "$run_dir/final_state.pth.tar" ]] && \
       ! pgrep -f -- "--exp-name ${exp_name}" >/dev/null; then
      printf '%s\n' "$run_dir/model_best.pth.tar"
      return
    fi
    sleep 120
  done
}

evaluate_model() {
  local name=$1 model=$2 geom=$3
  export HDF5_USE_FILE_LOCKING=FALSE
  export VFT_FULL_RANDOM_MASK=0 REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
  export RAY_LAMBDA=0 STUDENT_RAY_W=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0
  export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=$geom GBT_VIEW_AWARE=1
  export GBT_V2_SCALE=0 GBT_V3_SCALE=0 GBT_TOKEN_DROPOUT=0
  for k in 2 3 4; do
    CUDA_VISIBLE_DEVICES=1 "$EVAL" "$model" "${name}_v${k}" "$k" \
      "/mnt/data/cjyoutput/cmu_v${k}_eval_${name}_20260718" 0
  done
}

aux025=$(wait_for_model \
  distill_dualhard_legw09_gbt_k4plus_aux025_20260717 \
  distill_dualhard_legw09_gbt_k4plus_aux025_20260717)
aux05=$(wait_for_model \
  distill_dualhard_legw09_gbt_k4plus_aux05_20260717 \
  distill_dualhard_legw09_gbt_k4plus_aux05_20260717)
nogbt=$(wait_for_model \
  distill_dualhard_legw09_nogbt_aux025_20260717 \
  distill_dualhard_legw09_nogbt_aux025_20260717)

evaluate_model dualhard_gbt_aux025 "$aux025" 0.12
evaluate_model dualhard_gbt_aux05 "$aux05" 0.12
evaluate_model dualhard_nogbt_aux025 "$nogbt" 0

: > "$REPORT"
for name in dualhard_gbt_aux025 dualhard_gbt_aux05 dualhard_nogbt_aux025; do
  for k in 2 3 4; do
    {
      echo "=== ${name} V${k} ==="
      if [[ $k == 2 ]]; then
        python3 "$SUMMARY" "/mnt/data/cjyoutput/cmu_v2_eval_${name}_20260718" \
          --rumpl-values "$V2_RUMPL" \
          --baseline /mnt/data/cjyoutput/cmu_v2_eval_hardv_legw09_full_20260712_fg \
          --name "$name"
      elif [[ $k == 3 ]]; then
        python3 "$SUMMARY" "/mnt/data/cjyoutput/cmu_v3_eval_${name}_20260718" \
          --rumpl /mnt/data/cjyoutput/cmu_v3_eval_org_20260711 \
          --baseline /mnt/data/cjyoutput/cmu_v3_eval_hardv_legw09_full_20260714 \
          --name "$name"
      else
        python3 "$SUMMARY" "/mnt/data/cjyoutput/cmu_v4_eval_${name}_20260718" \
          --rumpl /mnt/data/cjyoutput/cmu_v4_eval_rumpl_conf_20260714 \
          --baseline /mnt/data/cjyoutput/cmu_v4_eval_hardv_legw09_full_20260714 \
          --name "$name"
      fi
    } >> "$REPORT"
  done
done
