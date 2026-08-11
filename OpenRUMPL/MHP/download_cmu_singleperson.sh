#!/usr/bin/env bash
# 极简温柔下载: 顺序一个一个下, 502/失败就等30s重试, 断点续传。
EP="http://domedb.perception.cs.cmu.edu/webdata/dataset"
DST=/mnt/data/cjydata/cmu_singleperson

get() {  # url out
  local url="$1" out="$2"
  while true; do
    if [ -f "$out" ] && [ ! -f "$out.aria2" ]; then
      echo "[$(date '+%H:%M:%S')] OK $(basename "$out") $(numfmt --to=iec $(stat -c%s "$out" 2>/dev/null||echo 0))"
      return
    fi
    aria2c -x8 -s8 -k1M --file-allocation=none --continue=true --max-tries=1 \
      --timeout=30 --connect-timeout=10 --console-log-level=error --summary-interval=0 \
      -d "$(dirname "$out")" -o "$(basename "$out")" "$url" >/dev/null 2>&1
    [ -f "$out" ] && [ ! -f "$out.aria2" ] && continue
    echo "[$(date '+%H:%M:%S')] retry $(basename "$out") ..."
    sleep 30
  done
}

for SEQ in 171204_pose5 171204_pose6; do
  mkdir -p "$DST/$SEQ/hdVideos"
  cp -n /mnt/data/cjydata/cmu_calibs/calibration_${SEQ}.json "$DST/$SEQ/" 2>/dev/null
  get "$EP/$SEQ/hdPose3d_stage1_coco19.tar" "$DST/$SEQ/hdPose3d_stage1_coco19.tar"
  for C in 03 06 12 13 23; do
    get "$EP/$SEQ/videos/hd_shared_crf20/hd_00_${C}.mp4" "$DST/$SEQ/hdVideos/hd_00_${C}.mp4"
  done
done
echo "[$(date '+%H:%M:%S')] === 全部完成 ==="
du -sh "$DST"/* 2>/dev/null
