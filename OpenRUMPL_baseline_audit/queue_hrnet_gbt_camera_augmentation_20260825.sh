#!/usr/bin/env bash
set -euo pipefail

H18_ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/best_available_modules_20260825_safe_candidates/hrnet/canonical_h18
# The continuous-time H18 queue includes both the current formal run and its
# generalization continuation. The durable COMPLETED marker is the hand-off;
# no transient process identifier is used here.
while [[ ! -f "${H18_ROOT}/model_generalization_v2_continuous_timewarp/COMPLETED" ]]; do
  sleep 30
done

# Generator continuation needs essentially the whole physical GPU0. Require
# three consecutive low-memory checks to avoid racing a process that is just
# starting or releasing memory.
stable=0
while (( stable < 3 )); do
  used=$(nvidia-smi --id=0 --query-compute-apps=used_memory \
    --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')
  if (( used <= 512 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  echo "gpu0_used_mib=${used} stable_checks=${stable}/3"
  (( stable >= 3 )) || sleep 30
done

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_hrnet_gbt_camera_augmentation_20260825.sh
