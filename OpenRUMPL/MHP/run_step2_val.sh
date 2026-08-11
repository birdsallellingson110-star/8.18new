#!/usr/bin/env bash
# Step2 validation 双卡并行调度器: 8 worker, 每卡 4 个, 处理 stage_IV/validation split 0..98
NWORKERS=8
NSPLITS=99
LOGDIR=/mnt/data/cjydata/step2_logs_val
mkdir -p "$LOGDIR"

run_worker() {
  local w=$1 gpu=$((w % 2))
  for ((s=w; s<NSPLITS; s+=NWORKERS)); do
    echo "[val w$w][gpu $gpu] >>> split $s 开始 $(date '+%F %T')"
    bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step2_split.sh "$s" "$gpu" validation \
      >> "$LOGDIR/split_${s}.log" 2>&1
    echo "[val w$w][gpu $gpu] <<< split $s 结束(exit $?) $(date '+%F %T')"
  done
}

echo "=== Step2 VAL 调度启动: $NWORKERS workers $(date '+%F %T') ==="
for ((w=0; w<NWORKERS; w++)); do run_worker "$w" >> "$LOGDIR/worker_${w}.log" 2>&1 & done
wait
echo "=== Step2 VAL 并行阶段结束, 检查/补跑缺失 split (防 OOM 掉的) $(date '+%F %T') ==="

# 串行补跑任何缺失的 split (单进程独占 GPU, 满显存, 不会再 OOM)
SV=/mnt/data/cjydata/mhp_workspace/paper_single_cmu/stage_V/validation
for pass in 1 2; do
  for ((s=0; s<NSPLITS; s++)); do
    if ! ls "$SV"/split_${s}_*.pkl >/dev/null 2>&1; then
      echo "[retry pass$pass] 补跑缺失 val split $s $(date '+%F %T')"
      bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step2_split.sh "$s" 0 validation \
        >> "$LOGDIR/split_${s}_retry.log" 2>&1
    fi
  done
done
echo "=== Step2 VAL 全部完成 $(date '+%F %T') ==="
