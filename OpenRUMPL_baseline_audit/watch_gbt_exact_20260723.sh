#!/usr/bin/env bash
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
run_root=/mnt/data/cjyoutput/baseline_reaudit_20260722
output_root="$run_root/output/multiview_amass_rumpl/multiview_rumpl_999"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
log="$run_root/watch_gbt_exact_20260723.log"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0
export RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=1
export GBT_USE_CONF_BIAS=1
export GBT_USE_GEOM_BIAS=1
export GBT_CONF_INIT=0.1

exec > >(tee -a "$log") 2>&1
echo "WATCH_START $(date --iso-8601=seconds)"

while tmux has-session -t rumpl_gbt_formula_exact_20260723 2>/dev/null || \
      tmux has-session -t rumpl_gbt_fusion_exact_20260723 2>/dev/null; do
  echo "WAIT $(date --iso-8601=seconds)"
  sleep 60
done

status=0
for spec in \
  "G0_gbt_formula_exact_seed0_20260723:0:1.0:formula" \
  "G1_gbt_fusion_exact_seed0_20260723:1:0.1:fusion"; do
  IFS=: read -r variant fusion_geom geom_init tag <<< "$spec"
  output_dir=$(find "$output_root" -maxdepth 1 -type d -name "${variant}_*" -print | sort | tail -n 1)
  if [[ -z "$output_dir" ]]; then
    echo "ERROR $variant output directory missing"
    status=1
    continue
  fi
  epoch=$(cat "$output_dir/epoch.txt" 2>/dev/null || true)
  if [[ "$epoch" != "20" || ! -f "$output_dir/final_state.pth.tar" || ! -f "$output_dir/model_best.pth.tar" ]]; then
    echo "ERROR $variant incomplete epoch=${epoch:-missing}"
    status=1
    continue
  fi

  final_prediction="$output_dir/preds_gt_multiview_cmu_panoptic_rumpl_mmpose__dict.pkl"
  "$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" "$final_prediction" \
    --output-json "$run_root/${variant}_final_epoch20_summary.json" || status=1

  export GBT_GEOM_INIT="$geom_init"
  export GBT_FUSION_GEOM="$fusion_geom"
  eval_root="$run_root/model_best_eval/$tag"
  eval_log="$run_root/${variant}_model_best_eval.log"
  mkdir -p "$eval_root" "$run_root/model_best_eval/log_$tag"
  (
    cd "$repo/RUMPL" || exit 1
    "$python" run/valid_rumpl.py \
      --cfg "$cfg" --gpus 0 --workers 16 \
      --model-file "$output_dir/model_best.pth.tar" \
      --modelDir "$eval_root" --logDir "$run_root/model_best_eval/log_$tag" \
      --state "best_$tag" --eval-comments "exact_best_$tag" --use-mmpose-val
  ) > "$eval_log" 2>&1 || {
    echo "ERROR $variant model_best evaluation failed"
    status=1
    continue
  }
  best_prediction=$(find "$eval_root" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
  "$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" "$best_prediction" \
    --output-json "$run_root/${variant}_model_best_summary.json" || status=1
  sha256sum "$output_dir/final_state.pth.tar" "$output_dir/model_best.pth.tar" \
    "$run_root/${variant}_final_epoch20_summary.json" \
    "$run_root/${variant}_model_best_summary.json"
done

echo "WATCH_END $(date --iso-8601=seconds) status=$status"
exit "$status"
