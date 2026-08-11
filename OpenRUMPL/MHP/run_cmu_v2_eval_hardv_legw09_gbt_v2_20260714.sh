#!/usr/bin/env bash
set -euo pipefail

cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=configs/cmu_panoptic/rumpl_amass/cmu_eval_sp_v2_conf.yaml
MODEL=/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/distill_hardv_legw09_gbt_v2_20260712_2026-07-12_16-29-02/model_best.pth.tar
COMMENT_PREFIX=hardv_legw09_gbt_v2
OUT_ROOT=${1:-/mnt/data/cjyoutput/cmu_v2_eval_hardv_legw09_gbt_v2_20260714}

export GBT_CONF_BIAS=0.35
export GBT_GEOM_BIAS=0.15
export GBT_VIEW_AWARE=1
export GBT_V2_SCALE=1.0

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
echo "[gbt] conf=$GBT_CONF_BIAS geom=$GBT_GEOM_BIAS view_aware=$GBT_VIEW_AWARE v2_scale=$GBT_V2_SCALE"

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
    --eval-comments "${COMMENT_PREFIX}_${tag}" \
    >"$log" 2>&1
  echo "[done] $(date '+%F %T') views=$combo"
done

echo "[finish] $(date '+%F %T')"
