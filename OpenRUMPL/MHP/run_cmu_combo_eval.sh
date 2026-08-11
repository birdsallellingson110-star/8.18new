#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 MODEL COMMENT_PREFIX K OUT_ROOT" >&2
  exit 2
fi

MODEL=$1
COMMENT_PREFIX=$2
K=$3
OUT_ROOT=$4

cd /home/lixiaob/cjy/OpenRUMPL/RUMPL
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=configs/cmu_panoptic/rumpl_amass/cmu_eval_sp_v2_conf.yaml

mkdir -p "$OUT_ROOT"

if [[ "$K" == "3" ]]; then
  COMBOS=(
    "3 6 12"
    "3 6 13"
    "3 6 23"
    "3 12 13"
    "3 12 23"
    "3 13 23"
    "6 12 13"
    "6 12 23"
    "6 13 23"
    "12 13 23"
  )
elif [[ "$K" == "4" ]]; then
  COMBOS=(
    "3 6 12 13"
    "3 6 12 23"
    "3 6 13 23"
    "3 12 13 23"
    "6 12 13 23"
  )
else
  echo "Unsupported K=$K; expected 3 or 4" >&2
  exit 2
fi

echo "[start] $(date '+%F %T')"
echo "[model] $MODEL"
echo "[prefix] $COMMENT_PREFIX"
echo "[k] $K"
echo "[out] $OUT_ROOT"

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
