#!/usr/bin/env bash
# 并发=3 下载器: 同时下3个文件(卡住一个不影响另俩), 断点续传, 502自动重试。
# 比单线程快~3倍, 又比10并发温柔(不打崩服务器)。
EP="http://domedb.perception.cs.cmu.edu/webdata/dataset"
DST=/mnt/data/cjydata/cmu_singleperson
PAR=3

fetch() {  # url out
  local url="$1" out="$2"
  while true; do
    if [ -f "$out" ] && [ ! -f "$out.aria2" ]; then
      echo "[$(date '+%H:%M:%S')] DONE $(basename "$out")"; return
    fi
    aria2c -x6 -s6 -k1M --file-allocation=none --continue=true --max-tries=1 \
      --timeout=30 --connect-timeout=10 --console-log-level=error --summary-interval=0 \
      -d "$(dirname "$out")" -o "$(basename "$out")" "$url" >/dev/null 2>&1
    [ -f "$out" ] && [ ! -f "$out.aria2" ] && continue
    sleep 12
  done
}

JOBS=()
for SEQ in 171204_pose5 171204_pose6; do
  mkdir -p "$DST/$SEQ/hdVideos"
  cp -n /mnt/data/cjydata/cmu_calibs/calibration_${SEQ}.json "$DST/$SEQ/" 2>/dev/null
  JOBS+=("$EP/$SEQ/hdPose3d_stage1_coco19.tar|$DST/$SEQ/hdPose3d_stage1_coco19.tar")
  for C in 03 06 12 13 23; do
    JOBS+=("$EP/$SEQ/videos/hd_shared_crf20/hd_00_${C}.mp4|$DST/$SEQ/hdVideos/hd_00_${C}.mp4")
  done
done

for job in "${JOBS[@]}"; do
  fetch "${job%%|*}" "${job##*|}" &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
done
wait
echo "[$(date '+%H:%M:%S')] === 全部完成 ==="
du -sh "$DST"/171204_pose5 "$DST"/171204_pose6 2>/dev/null
