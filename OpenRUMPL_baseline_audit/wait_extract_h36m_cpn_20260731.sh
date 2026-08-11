#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cjydata/datasets/h36m_external_2d_cpn
ARCHIVE=${ROOT}/h36m_videopose3d_processed_download
EXTRACTED=${ROOT}/extracted

while [[ ! -s "${ARCHIVE}" ]]; do
  sleep 20
done

mkdir -p "${EXTRACTED}"
/home/lixiaob/miniconda3/bin/bsdtar -tf "${ARCHIVE}" \
  >"${ROOT}/archive_manifest.txt"
if ! find "${EXTRACTED}" -type f -name '*.npz' -print -quit \
  | grep -q .; then
  /home/lixiaob/miniconda3/bin/bsdtar -xf "${ARCHIVE}" -C "${EXTRACTED}"
fi

find "${EXTRACTED}" -type f -printf '%p %s bytes\n' \
  | sort >"${ROOT}/extracted_manifest.txt"
echo "[CPN extract] complete $(date --iso-8601=seconds)"
