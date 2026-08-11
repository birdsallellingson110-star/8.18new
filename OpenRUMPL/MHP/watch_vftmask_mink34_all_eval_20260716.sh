#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
RUN3=$ROOT/distill_hardv_legw09_vftmask02_mink3_20260716_2026-07-16_18-46-40
RUN4=$ROOT/distill_hardv_legw09_vftmask02_mink4_20260716_2026-07-16_18-45-23
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh

echo "[watch] $(date '+%F %T') waiting for minK=3 and minK=4 final checkpoints"
while [[ ! -f "$RUN3/final_state.pth.tar" || ! -f "$RUN4/final_state.pth.tar" ]]; do
  sleep 300
done
while pgrep -f 'run/train_rumpl.py.*distill_hardv_legw09_vftmask02_mink[34]_20260716' >/dev/null; do
  sleep 60
done

export VFT_FULL_RANDOM_MASK=0
export VFT_MASK_MIN_VIEWS=2
export CAA_LAMBDA=0
export DEPRO_LAMBDA=0
export GBT_CONF_BIAS=0
export GBT_GEOM_BIAS=0
export GBT_VIEW_AWARE=0
export GBT_V2_SCALE=0
export GBT_TOKEN_DROPOUT=0

eval_model() {
  local run=$1
  local tag=$2
  local gpu=$3
  local k
  for k in 2 3 4; do
    echo "[eval] $(date '+%F %T') $tag V$k gpu=$gpu"
    "$EVAL" \
      "$run/model_best.pth.tar" \
      "${tag}_v${k}" \
      "$k" \
      "/mnt/data/cjyoutput/cmu_v${k}_eval_${tag}_20260717" \
      "$gpu"
  done
}

eval_model "$RUN3" hardv_legw09_vftmask02_mink3 0 &
pid3=$!
eval_model "$RUN4" hardv_legw09_vftmask02_mink4 1 &
pid4=$!
wait "$pid3"
wait "$pid4"
echo "[finish] $(date '+%F %T') all minK=3/minK=4 evaluations complete"
