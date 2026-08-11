#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {random2to4|fixed2} PHYSICAL_GPU" >&2
  exit 2
fi

# Let the short four-way validation export finish first, then use both GPUs
# for the clean original-RUMPL ablation while the long training export keeps
# streaming in the background.
while tmux has-session -t cjy_a0_heatmap_val_dense 2>/dev/null; do
  sleep 30
done

exec bash \
  /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_H20_original_rumpl_tri_anchor_20260731.sh \
  "$1" "$2"
