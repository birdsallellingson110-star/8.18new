#!/usr/bin/env bash
set -uo pipefail

# CMU Panoptic single-person pose protocol:
# train: 171026_pose1-3, 171204_pose1-4
# test:  171204_pose5-6
#
# The five standard evaluation cameras are 3, 6, 12, 13, 23.  The remaining
# cameras were selected by farthest-point sampling over the normalized HD
# camera centers in calibration_171204_pose5.json, seeded by those five.

CMU_ENDPOINT="http://domedb.perception.cs.cmu.edu/webdata/dataset"
SNU_ENDPOINT="http://vcl.snu.ac.kr/panoptic/webdata/dataset"
DESTINATION="/mnt/data/cjydata/cmu_singleperson_real20"
EXISTING_ROOT="/mnt/data/cjydata/cmu_singleperson"

# Short training sequences come first so a complete full-frame temporal subset
# becomes usable as early as possible.
TRAIN_SEQUENCES=(
  171026_pose3 171204_pose3 171026_pose2 171026_pose1
  171204_pose4 171204_pose1 171204_pose2
)

TEST_SEQUENCES=(
  171204_pose5 171204_pose6
)

SEQUENCES=(
  "${TRAIN_SEQUENCES[@]}"
  "${TEST_SEQUENCES[@]}"
)

STANDARD_CAMERAS=(
  03 06 12 13 23
)

EXTRA_CAMERAS=(
  01 02 04 05 07 10 16 17 18 19 22 24 25 27 28
)

CAMERAS=(
  "${STANDARD_CAMERAS[@]}"
  "${EXTRA_CAMERAS[@]}"
)

mkdir -p "$DESTINATION"

choose_endpoint() {
  # The official toolbox added the SNU mirror in 2024.  Probe it without the
  # machine's HTTP proxy before every new file, so a recovered mirror is used
  # automatically.  The tiny calibration file also prevents accepting an HTTP
  # error page as a valid mirror response.
  if curl \
    --noproxy '*' \
    --fail \
    --location \
    --silent \
    --show-error \
    --connect-timeout 3 \
    --max-time 6 \
    --output /dev/null \
    "$SNU_ENDPOINT/171026_pose1/calibration_171026_pose1.json"; then
    printf '%s\n' "$SNU_ENDPOINT"
  else
    printf '%s\n' "$CMU_ENDPOINT"
  fi
}

download_one() {
  local remote_path="$1"
  local output="$2"
  local output_dir output_name endpoint url
  local snu_failed=0
  output_dir="$(dirname "$output")"
  output_name="$(basename "$output")"
  mkdir -p "$output_dir"

  while true; do
    if [[ -s "$output" && ! -e "$output.aria2" ]]; then
      echo "[$(date '+%F %T')] present $output"
      return 0
    fi

    if [[ "$snu_failed" -eq 0 ]]; then
      endpoint="$(choose_endpoint)"
    else
      endpoint="$CMU_ENDPOINT"
    fi
    url="$endpoint/$remote_path"
    echo "[$(date '+%F %T')] download $url"
    aria2c \
      --continue=true \
      --file-allocation=none \
      --max-connection-per-server=4 \
      --split=4 \
      --min-split-size=16M \
      --connect-timeout=20 \
      --timeout=60 \
      --max-tries=5 \
      --retry-wait=10 \
      --auto-file-renaming=false \
      --allow-overwrite=true \
      --console-log-level=warn \
      --summary-interval=60 \
      --dir="$output_dir" \
      --out="$output_name" \
      "$url"

    if [[ -s "$output" && ! -e "$output.aria2" ]]; then
      echo "[$(date '+%F %T')] complete $output $(numfmt --to=iec "$(stat -c%s "$output")")"
      return 0
    fi

    if [[ "$endpoint" == "$SNU_ENDPOINT" ]]; then
      snu_failed=1
      echo "[$(date '+%F %T')] SNU item failed; falling back to CMU"
    fi
    echo "[$(date '+%F %T')] retry in 30s $output"
    sleep 30
  done
}

echo "[$(date '+%F %T')] destination=$DESTINATION"
echo "[$(date '+%F %T')] sequences=${SEQUENCES[*]}"
echo "[$(date '+%F %T')] cameras=${CAMERAS[*]}"
df -h "$DESTINATION"

# Reuse the already downloaded standard-camera pose5/pose6 assets without
# duplicating tens of GB on the mounted disk.
for sequence in 171204_pose5 171204_pose6; do
  mkdir -p "$DESTINATION/$sequence/hdVideos"
  for camera in 03 06 12 13 23; do
    source_video="$EXISTING_ROOT/$sequence/hdVideos/hd_00_${camera}.mp4"
    target_video="$DESTINATION/$sequence/hdVideos/hd_00_${camera}.mp4"
    if [[ -s "$source_video" && ! -e "$target_video" ]]; then
      ln -s "$source_video" "$target_video"
    fi
  done
done

# Small metadata first.
for sequence in "${SEQUENCES[@]}"; do
  download_one \
    "$sequence/calibration_${sequence}.json" \
    "$DESTINATION/$sequence/calibration_${sequence}.json"
  download_one \
    "$sequence/hdPose3d_stage1_coco19.tar" \
    "$DESTINATION/$sequence/hdPose3d_stage1_coco19.tar"
done

# Measured on this server on 2026-07-27:
#   one file x one range:       about 154 KiB/s
#   one file x four ranges:     about 201 KiB/s
#   two files x four ranges:    about  61 KiB/s aggregate
#   sixteen concurrent files:   about  67 KiB/s aggregate
# CMU throttles high aggregate concurrency, so download exactly one video at a
# time with four ranges.  This is roughly 3x faster than the original 4x4 plan.

# Phase 1: all full-rate frames from the standard five cameras.  This produces
# the first usable temporal training set as early as possible.
for sequence in "${TRAIN_SEQUENCES[@]}"; do
  for camera in "${STANDARD_CAMERAS[@]}"; do
    download_one \
      "$sequence/videos/hd_shared_crf20/hd_00_${camera}.mp4" \
      "$DESTINATION/$sequence/hdVideos/hd_00_${camera}.mp4"
  done
done
touch "$DESTINATION/TRAIN_STANDARD5_COMPLETE"

# Phase 2: expand all seven training sequences to the full selected 20-camera
# pool, preserving every video frame.
for sequence in "${TRAIN_SEQUENCES[@]}"; do
  for camera in "${EXTRA_CAMERAS[@]}"; do
    download_one \
      "$sequence/videos/hd_shared_crf20/hd_00_${camera}.mp4" \
      "$DESTINATION/$sequence/hdVideos/hd_00_${camera}.mp4"
  done
done
touch "$DESTINATION/TRAIN_REAL20_COMPLETE"

# Phase 3: the official five test cameras already exist locally.  Download the
# extra fifteen test cameras last; they are useful for expanded camera studies
# but do not block standard pose5/pose6 evaluation.
for sequence in "${TEST_SEQUENCES[@]}"; do
  for camera in "${EXTRA_CAMERAS[@]}"; do
    download_one \
      "$sequence/videos/hd_shared_crf20/hd_00_${camera}.mp4" \
      "$DESTINATION/$sequence/hdVideos/hd_00_${camera}.mp4"
  done
done

echo "[$(date '+%F %T')] validating file counts"
failed=0
for sequence in "${SEQUENCES[@]}"; do
  count="$(find "$DESTINATION/$sequence/hdVideos" -maxdepth 1 \( -type f -o -type l \) | wc -l)"
  echo "$sequence videos=$count"
  if [[ "$count" -ne 20 ]]; then
    failed=1
  fi

  if [[ ! -s "$DESTINATION/$sequence/calibration_${sequence}.json" ]]; then
    failed=1
  fi
  if [[ ! -s "$DESTINATION/$sequence/hdPose3d_stage1_coco19.tar" ]]; then
    failed=1
  fi
done

du -sh "$DESTINATION"
df -h "$DESTINATION"

if [[ "$failed" -ne 0 ]]; then
  echo "[$(date '+%F %T')] validation failed"
  exit 1
fi

touch "$DESTINATION/DOWNLOAD_COMPLETE"
echo "[$(date '+%F %T')] all 9 sequences and 20 cameras complete"
