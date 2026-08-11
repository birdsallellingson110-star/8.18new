#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU VARIANT FUSION_GEOM GEOM_INIT [USE_CONF] [USE_GEOM] [CONF_INIT]}
variant=${2:?usage: $0 GPU VARIANT FUSION_GEOM GEOM_INIT [USE_CONF] [USE_GEOM] [CONF_INIT]}
fusion_geom=${3:?usage: $0 GPU VARIANT FUSION_GEOM GEOM_INIT [USE_CONF] [USE_GEOM] [CONF_INIT]}
geom_init=${4:?usage: $0 GPU VARIANT FUSION_GEOM GEOM_INIT [USE_CONF] [USE_GEOM] [CONF_INIT]}
use_conf=${5:-1}
use_geom=${6:-1}
conf_init=${7:-0.1}

if [[ "$fusion_geom" != "0" && "$fusion_geom" != "1" ]]; then
  echo "FUSION_GEOM must be 0 or 1" >&2
  exit 2
fi
if [[ "$use_conf" != "0" && "$use_conf" != "1" ]]; then
  echo "USE_CONF must be 0 or 1" >&2
  exit 2
fi
if [[ "$use_geom" != "0" && "$use_geom" != "1" ]]; then
  echo "USE_GEOM must be 0 or 1" >&2
  exit 2
fi
if [[ "$use_conf" == "0" && "$use_geom" == "0" ]]; then
  echo "At least one bias must be enabled" >&2
  exit 2
fi

export GBT_LEARNABLE_BIAS=1
export GBT_USE_CONF_BIAS="$use_conf"
export GBT_USE_GEOM_BIAS="$use_geom"
export GBT_CONF_INIT="$conf_init"
export GBT_GEOM_INIT="$geom_init"
export GBT_FUSION_GEOM="$fusion_geom"

# Preserve the closest baseline protocol: public PFT behavior, corrected scheduler, 16 workers.
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 16
