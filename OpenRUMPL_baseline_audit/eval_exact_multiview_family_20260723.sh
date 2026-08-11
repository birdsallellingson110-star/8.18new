#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU FAMILY CHECKPOINT MODE}
family=${2:?usage: $0 GPU FAMILY CHECKPOINT MODE}
checkpoint=${3:?usage: $0 GPU FAMILY CHECKPOINT MODE}
mode=${4:?usage: $0 GPU FAMILY CHECKPOINT MODE}

for n_views in 2 3 4 5; do
  /home/lixiaob/cjy/OpenRUMPL_baseline_audit/eval_exact_multiview_20260723.sh \
    "$gpu" "${family}_v${n_views}" "$checkpoint" "$n_views" "$mode"
done
