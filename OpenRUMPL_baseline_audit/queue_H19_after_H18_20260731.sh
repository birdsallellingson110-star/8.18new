#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {center_nodrop|center_drop20} GPU H18_DONE_FILE" >&2
  exit 2
fi

variant=$1
gpu=$2
prerequisite=$3
launcher=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H18_paper_aligned_singleframe_20260731.sh

while [[ ! -s "${prerequisite}" ]]; do
  sleep 30
done

exec bash "${launcher}" "${variant}" "${gpu}"
