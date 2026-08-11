#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
RUN02=$ROOT/distill_hardv_legw09_vftmask02_20260716_2026-07-16_01-10-52
RUN05=$ROOT/distill_hardv_legw09_vftmask05_20260716_2026-07-16_01-17-28
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh

echo "[watch] $(date '+%F %T') waiting for both final checkpoints"
while [[ ! -f "$RUN02/final_state.pth.tar" || ! -f "$RUN05/final_state.pth.tar" ]]; do
  sleep 300
done

# final_state is written at training completion; also wait for CUDA contexts to exit.
while pgrep -f 'run/train_rumpl.py.*distill_hardv_legw09_vftmask0[25]_20260716' >/dev/null; do
  sleep 60
done

export VFT_FULL_RANDOM_MASK=0
export CAA_LAMBDA=0
export DEPRO_LAMBDA=0
export GBT_CONF_BIAS=0
export GBT_GEOM_BIAS=0
export GBT_VIEW_AWARE=0
export GBT_V2_SCALE=0
export GBT_TOKEN_DROPOUT=0

echo "[eval] $(date '+%F %T') M=0.2 V2"
"$EVAL" \
  "$RUN02/model_best.pth.tar" \
  hardv_legw09_vftmask02_v2 \
  2 \
  /mnt/data/cjyoutput/cmu_v2_eval_hardv_legw09_vftmask02_20260716 \
  1

echo "[eval] $(date '+%F %T') M=0.5 V2"
"$EVAL" \
  "$RUN05/model_best.pth.tar" \
  hardv_legw09_vftmask05_v2 \
  2 \
  /mnt/data/cjyoutput/cmu_v2_eval_hardv_legw09_vftmask05_20260716 \
  1

echo "[finish] $(date '+%F %T') both V2 evaluations complete"
