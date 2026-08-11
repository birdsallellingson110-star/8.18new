#!/usr/bin/env bash
set -euo pipefail

log=/mnt/data/cjyoutput/temporal_exact_20260723/queue.log
mkdir -p /mnt/data/cjyoutput/temporal_exact_20260723
exec > >(tee -a "$log") 2>&1
echo "QUEUE_START $(date --iso-8601=seconds)"

while tmux has-session -t rumpl_gbt_conf_ablation_20260723 2>/dev/null || \
      tmux has-session -t rumpl_gbt_geom_ablation_20260723 2>/dev/null || \
      tmux has-session -t rumpl_gbt_ablation_watcher_20260723 2>/dev/null; do
  echo "WAIT_GBT $(date --iso-8601=seconds)"
  sleep 60
done

tmux new-session -d -s rumpl_temporal_natural_20260723 \
  '/home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_exact_temporal_20260723.sh 0 T0_r5_postvft_l9_natural_seed0_20260723 0.0'
tmux new-session -d -s rumpl_temporal_clean_20260723 \
  '/home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_exact_temporal_20260723.sh 1 T1_r5_postvft_l9_oks05_seed0_20260723 0.5'
echo "LAUNCHED $(date --iso-8601=seconds)"

while tmux has-session -t rumpl_temporal_natural_20260723 2>/dev/null || \
      tmux has-session -t rumpl_temporal_clean_20260723 2>/dev/null; do
  echo "WAIT_TEMPORAL $(date --iso-8601=seconds)"
  sleep 60
done
echo "QUEUE_END $(date --iso-8601=seconds)"
