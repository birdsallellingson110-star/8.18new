#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT MODE}
tag=${2:?usage: $0 GPU TAG CHECKPOINT MODE}
checkpoint=${3:?usage: $0 GPU TAG CHECKPOINT MODE}
mode=${4:?usage: $0 GPU TAG CHECKPOINT MODE}

for n_views in 2 3 4 5; do
  /home/lixiaob/cjy/OpenRUMPL_baseline_audit/eval_exact_gbt_single_20260724.sh \
    "$gpu" "${tag}_v${n_views}" "$checkpoint" "$n_views" "$mode"
done
