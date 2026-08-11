#!/usr/bin/env bash
set -euo pipefail

# H19 currently owns the two GPUs.  Start the full validation heatmap export as
# soon as both H19 train/evaluate launchers have released them.  The training
# heatmap exporter is intentionally allowed to continue in parallel.
while tmux has-session -t cjy_h19_p2 2>/dev/null \
   || tmux has-session -t cjy_h19_p3 2>/dev/null; do
  sleep 30
done

exec bash \
  /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_A0_h36m_val_heatmap_topk_20260731.sh
