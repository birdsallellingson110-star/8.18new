#!/usr/bin/env bash
set -euo pipefail

pcs_bin="/home/lixiaob/cjy/tools/BaiduPCS-Go-v4.0.1/BaiduPCS-Go-v4.0.1-linux-amd64/BaiduPCS-Go"
export BAIDUPCS_GO_CONFIG_DIR="/mnt/data/cjydata/baidupcs_config_h36m"

printf '%s\n' "请粘贴百度网盘网页请求中的完整 Cookie，然后按回车。"
printf '%s\n' "输入不会回显，也不会写入 shell 历史："
IFS= read -r -s cookie_value
printf '\n'

if [[ -z "${cookie_value}" ]]; then
  printf '%s\n' "未输入 Cookie，已取消。"
  exit 1
fi

"${pcs_bin}" login -cookies="${cookie_value}"
unset cookie_value
"${pcs_bin}" who
