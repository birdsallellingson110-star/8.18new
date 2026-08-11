#!/usr/bin/env bash
set -euo pipefail

RUMPL=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFGDIR=$RUMPL/configs/cmu_panoptic/rumpl_amass
LOGDIR=/mnt/data/cjyoutput/resume_dualhard_k3plus_20260719
mkdir -p "$LOGDIR"

launch_train() {
  local name=$1 aux_w=$2 geom=$3
  local cfg=$CFGDIR/${name}.yaml
  local log=$LOGDIR/${name}.console.log

  nohup setsid bash -lc "
    cd '$RUMPL'
    source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1
    export CUDA_VISIBLE_DEVICES=1
    export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
    export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache PIP_CACHE_DIR=/mnt/data/cjydata/.pip_cache
    export MPLCONFIGDIR=/mnt/data/cjydata/.cache/matplotlib WANDB_MODE=offline
    export DISTILL_W=1 STUDENT_GT_W=1 STUDENT_VIEWS=2
    export HARD_VIEW_MINING=1 HARD_VIEW_CAND=3 LEG_DISTILL_W=0.9
    export AUX_MULTIK_W='$aux_w' AUX_MULTIK_MIN=3 AUX_MULTIK_MAX=4
    export GBT_CONF_BIAS=0 GBT_GEOM_BIAS='$geom' GBT_VIEW_AWARE=1
    export GBT_V2_SCALE=0 GBT_V3_SCALE=1 GBT_V4_SCALE=1 GBT_TOKEN_DROPOUT=0
    export VFT_FULL_RANDOM_MASK=0 REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
    export RAY_LAMBDA=0 STUDENT_RAY_W=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0
    exec '$PY' run/train_rumpl.py --cfg '$cfg' --gpus 0 --workers 8
  " >"$log" 2>&1 < /dev/null &
  echo "$! $name" | tee -a "$LOGDIR/pids.txt"
}

: > "$LOGDIR/pids.txt"
launch_train distill_dualhard_legw09_gbt_k3plus_aux05_20260718_2026-07-18_13-25-45 0.5 0.12
launch_train distill_dualhard_legw09_gbt_k3plus_aux075_20260718_2026-07-18_13-26-13 0.75 0.12
launch_train distill_dualhard_legw09_nogbt_aux05_20260718_2026-07-18_13-26-23 0.5 0

nohup setsid bash /home/lixiaob/cjy/OpenRUMPL/MHP/watch_dualhard_k3plus_eval_20260718.sh \
  >"$LOGDIR/watcher.console.log" 2>&1 < /dev/null &
echo "$! watcher" | tee -a "$LOGDIR/pids.txt"

