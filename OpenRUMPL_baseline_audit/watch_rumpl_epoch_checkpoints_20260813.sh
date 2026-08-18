#!/usr/bin/env bash
# Preserve every completed epoch from RUMPL's rolling checkpoint file.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_RUN_DIR DESTINATION_DIR" >&2
  exit 2
fi

source_dir=$1
destination_dir=$2
mkdir -p "${destination_dir}"
last=0

while true; do
  if [[ -s "${source_dir}/epoch.txt" && -s "${source_dir}/checkpoint.pth.tar" ]]; then
    epoch=$(tr -dc '0-9' < "${source_dir}/epoch.txt")
    if [[ -n "${epoch}" && "${epoch}" -gt "${last}" ]]; then
      target="${destination_dir}/checkpoint_epoch_${epoch}.pth.tar"
      temporary="${target}.tmp"
      if [[ ! -s "${target}" ]]; then
        cp --reflink=auto "${source_dir}/checkpoint.pth.tar" "${temporary}"
        mv "${temporary}" "${target}"
      fi
      sha256sum "${target}" > "${target}.sha256"
      printf '%s epoch=%s source=%s target=%s\n' \
        "$(date --iso-8601=seconds)" "${epoch}" "${source_dir}" "${target}"
      last=${epoch}
    fi
  fi
  if [[ -s "${source_dir}/final_state.pth.tar" ]]; then
    break
  fi
  sleep 5
done
