#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-1}
BASE=/mnt/data/cjyoutput/h36m_gbt_bias_20260729
LAUNCHER=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H14_h36m_gbt_bias_variant_20260729.sh
controller_log="${BASE}/H14_controller.log"

mkdir -p "${BASE}"
echo "[H14_CONTROLLER] start gpu=${gpu} $(date --iso-8601=seconds)" | tee "${controller_log}"
for mode in plain conf geom both baseline; do
  echo "[H14_CONTROLLER] launching ${mode} $(date --iso-8601=seconds)" | tee -a "${controller_log}"
  bash "${LAUNCHER}" "${mode}" "${gpu}" >>"${controller_log}" 2>&1
done
echo "[H14_CONTROLLER] complete $(date --iso-8601=seconds)" | tee -a "${controller_log}"
