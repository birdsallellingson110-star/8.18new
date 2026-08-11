#!/usr/bin/env bash
# 链式: 等 train 的 99 个 stage_V/train pkl 全部生成完 (GPU 空出来) 后,
# 自动开始 validation 的 step2 渲染。避免和 train 抢 GPU 显存。
set -uo pipefail
TRAIN_SV=/mnt/data/cjydata/mhp_workspace/paper_single_cmu/stage_V/train
LOG=/mnt/data/cjydata/step2_chain_val.log

echo "[chain] 等待 train 完成 (99 个 final pkl)... $(date '+%F %T')" >> "$LOG"
while true; do
  n=$(ls "$TRAIN_SV"/*.pkl 2>/dev/null | grep -v temp | wc -l)
  if [ "$n" -ge 99 ]; then break; fi
  sleep 120
done
echo "[chain] train 已完成 ($n pkl)。开始 validation step2 $(date '+%F %T')" >> "$LOG"

bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step2_val.sh >> "$LOG" 2>&1

vn=$(ls /mnt/data/cjydata/mhp_workspace/paper_single_cmu/stage_V/validation/*.pkl 2>/dev/null | grep -v temp | wc -l)
echo "[chain] validation step2 完成: $vn 个 pkl $(date '+%F %T')" >> "$LOG"
