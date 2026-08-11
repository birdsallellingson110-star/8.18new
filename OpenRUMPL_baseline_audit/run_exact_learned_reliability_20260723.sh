#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU VARIANT AUX_WEIGHT GATE_INIT TEMPERATURE}
variant=${2:?usage: $0 GPU VARIANT AUX_WEIGHT GATE_INIT TEMPERATURE}
aux_weight=${3:?usage: $0 GPU VARIANT AUX_WEIGHT GATE_INIT TEMPERATURE}
gate_init=${4:?usage: $0 GPU VARIANT AUX_WEIGHT GATE_INIT TEMPERATURE}
temperature=${5:?usage: $0 GPU VARIANT AUX_WEIGHT GATE_INIT TEMPERATURE [PAIR_GATE] [RANK_WEIGHT]}
pair_gate=${6:-0}
rank_weight=${7:-0.0}
workers=${8:-16}

export GBT_LEARNABLE_BIAS=0
export GBT_ORACLE_RELIABILITY=0
export GBT_LEARNED_RELIABILITY=1
export GBT_RELIABILITY_AUX_WEIGHT="$aux_weight"
export GBT_RELIABILITY_GATE_INIT="$gate_init"
export GBT_RELIABILITY_TEMPERATURE="$temperature"
export GBT_RELIABILITY_PAIR_GATE="$pair_gate"
export GBT_RELIABILITY_RANK_WEIGHT="$rank_weight"

exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 "$workers"
