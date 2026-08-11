#!/usr/bin/env bash
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
run_root=/mnt/data/cjyoutput/baseline_reaudit_20260722
output_root="$run_root/output/multiview_amass_rumpl/multiview_rumpl_999"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
log="$run_root/watch_scheduler_baselines.log"

exec > >(tee -a "$log") 2>&1
echo "WATCH_START $(date --iso-8601=seconds)"

while tmux has-session -t rumpl_r2_scheduler_20260722 2>/dev/null || \
      tmux has-session -t rumpl_r3_paper_clean_20260722 2>/dev/null; do
  echo "WAIT $(date --iso-8601=seconds)"
  sleep 60
done

status=0
for variant in R2_fix_scheduler_public_pft_exact_seed0_20260722 R3_fix_scheduler_and_pft_exact_seed0_20260722; do
  output_dir=$(find "$output_root" -maxdepth 1 -type d -name "${variant}_*" -print | sort | tail -n 1)
  if [[ -z "$output_dir" ]]; then
    echo "ERROR $variant output directory missing"
    status=1
    continue
  fi

  epoch=$(cat "$output_dir/epoch.txt" 2>/dev/null || true)
  if [[ "$epoch" != "20" || ! -f "$output_dir/final_state.pth.tar" ]]; then
    echo "ERROR $variant incomplete epoch=${epoch:-missing} final_state=$(test -f "$output_dir/final_state.pth.tar" && echo yes || echo no)"
    status=1
    continue
  fi

  prediction_file="$output_dir/preds_gt_multiview_cmu_panoptic_rumpl_mmpose__dict.pkl"
  summary_file="$run_root/${variant}_final_epoch20_summary.json"
  "$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
    "$prediction_file" --output-json "$summary_file"
  sha256sum "$output_dir/final_state.pth.tar" "$output_dir/model_best.pth.tar" "$summary_file"
done

echo "WATCH_END $(date --iso-8601=seconds) status=$status"
exit "$status"
