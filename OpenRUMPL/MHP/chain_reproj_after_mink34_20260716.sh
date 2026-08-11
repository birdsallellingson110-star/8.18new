#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999
EVAL=/home/lixiaob/cjy/OpenRUMPL/MHP/run_cmu_combo_eval_env_20260714.sh

echo "[wait] $(date '+%F %T') waiting for K-aware evaluation watcher"
while pgrep -f 'watch_vftmask_mink34_all_eval_20260716.sh' >/dev/null; do
  sleep 300
done

echo "[train] $(date '+%F %T') starting reprojection ablations"
/home/lixiaob/cjy/OpenRUMPL/MHP/run_distill_hardv_legw09_reproj_main02_20260717.sh \
  > /mnt/data/cjyoutput/distill_hardv_legw09_reproj_main02_20260717.launch.log 2>&1 &
pid_main=$!
/home/lixiaob/cjy/OpenRUMPL/MHP/run_distill_hardv_legw09_reproj_student02_20260717.sh \
  > /mnt/data/cjyoutput/distill_hardv_legw09_reproj_student02_20260717.launch.log 2>&1 &
pid_student=$!
wait "$pid_main"
wait "$pid_student"

run_main=$(find "$ROOT" -maxdepth 1 -type d -name 'distill_hardv_legw09_reproj_main02_20260717_*' | sort | tail -n 1)
run_student=$(find "$ROOT" -maxdepth 1 -type d -name 'distill_hardv_legw09_reproj_student02_20260717_*' | sort | tail -n 1)
[[ -f "$run_main/model_best.pth.tar" ]]
[[ -f "$run_student/model_best.pth.tar" ]]

export VFT_FULL_RANDOM_MASK=0 VFT_MASK_MIN_VIEWS=2
export REPROJ_LAMBDA=0 STUDENT_REPROJ_W=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0
export GBT_CONF_BIAS=0 GBT_GEOM_BIAS=0 GBT_VIEW_AWARE=0 GBT_V2_SCALE=0 GBT_TOKEN_DROPOUT=0

eval_model() {
  local run=$1 tag=$2 gpu=$3 k
  for k in 2 3 4; do
    echo "[eval] $(date '+%F %T') $tag V$k gpu=$gpu"
    "$EVAL" "$run/model_best.pth.tar" "${tag}_v${k}" "$k" \
      "/mnt/data/cjyoutput/cmu_v${k}_eval_${tag}_20260718" "$gpu"
  done
}

eval_model "$run_main" hardv_legw09_reproj_main02 0 &
eval_main=$!
eval_model "$run_student" hardv_legw09_reproj_student02 1 &
eval_student=$!
wait "$eval_main"
wait "$eval_student"
echo "[finish] $(date '+%F %T') reprojection train/eval chain complete"
