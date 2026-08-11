#!/usr/bin/env bash
set -euo pipefail

cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=configs/cmu_panoptic/rumpl_amass/cmu_eval_sp_v2_conf.yaml
MODEL=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/distill_hardv_debug_2026-07-10_14-31-32/model_best.pth.tar
OUT_ROOT=${1:-/mnt/data/cjyoutput/cmu_v2_eval_hardv_debug_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUT_ROOT"

declare -a COMBOS=(
  "3 6"
  "3 12"
  "3 13"
  "3 23"
  "6 12"
  "6 13"
  "6 23"
  "12 13"
  "12 23"
  "13 23"
)

echo "[start] $(date '+%F %T')"
echo "[out] $OUT_ROOT"
echo "[model] $MODEL"

for combo in "${COMBOS[@]}"; do
  tag=${combo// /_}
  log="$OUT_ROOT/views_${tag}.log"
  echo "[run] $(date '+%F %T') views=$combo log=$log"
  CUDA_VISIBLE_DEVICES=0 "$PY" run/valid_rumpl.py \
    --cfg "$CFG" \
    --gpus 0 \
    --workers 4 \
    --model-file "$MODEL" \
    --use-mmpose-val \
    --views $combo \
    --eval-comments "hardv_debug_v2_${tag}" \
    >"$log" 2>&1
  echo "[done] $(date '+%F %T') views=$combo"
done

echo "[finish] $(date '+%F %T')"
