#!/usr/bin/env bash
set -euo pipefail

pcs_bin="/home/lixiaob/cjy/tools/BaiduPCS-Go-v4.0.1/BaiduPCS-Go-v4.0.1-linux-amd64/BaiduPCS-Go"
export BAIDUPCS_GO_CONFIG_DIR="/mnt/data/cjydata/baidupcs_config_h36m"

save_dir="/mnt/data/cjydata/datasets/h36m_rumpl_official/downloads"
log_dir="/mnt/data/cjyoutput/h36m_download_20260727"
mkdir -p "${save_dir}" "${log_dir}"

"${pcs_bin}" download \
  "/我的资源/human3.6" \
  --saveto "${save_dir}" \
  --mode locate \
  --status \
  -p 8 \
  -l 3 \
  --retry 20 \
  --mtime
