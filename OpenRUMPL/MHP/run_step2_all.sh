#!/usr/bin/env bash
# Step2 双卡并行调度器: 8 个 worker, 每卡 4 个, 处理 stage_IV split 0..98
# worker w -> GPU (w%2); 处理 split 满足 (split % NWORKERS == w)
# 已生成 final pkl 的 split 会被脚本自动跳过(可断点续跑)。
NWORKERS=8
NSPLITS=99   # split 0..98
LOGDIR=/mnt/data/cjydata/step2_logs
mkdir -p "$LOGDIR"

run_worker() {
  local w=$1 gpu=$((w % 2))
  for ((s=w; s<NSPLITS; s+=NWORKERS)); do
    echo "[worker $w][gpu $gpu] >>> split $s 开始 $(date '+%F %T')"
    bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step2_split.sh "$s" "$gpu" \
      >> "$LOGDIR/split_${s}.log" 2>&1
    echo "[worker $w][gpu $gpu] <<< split $s 结束(exit $?) $(date '+%F %T')"
  done
  echo "[worker $w] 全部完成 $(date '+%F %T')"
}

echo "=== Step2 调度启动: $NWORKERS workers, splits 0..$((NSPLITS-1)) $(date '+%F %T') ==="
for ((w=0; w<NWORKERS; w++)); do
  run_worker "$w" >> "$LOGDIR/worker_${w}.log" 2>&1 &
done
wait
echo "=== Step2 全部 split 完成 $(date '+%F %T') ==="
