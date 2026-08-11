#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/temporal_exact_20260723
variant=${1:-T2_r5_postvft_l9_oks05_grouped_allviews_seed0_20260723}
checkpoint="$root/$variant/model_best.pth.tar"
annotation=/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched/cmu_panoptic_validation.pkl
output="$root/$variant/cmu_all_combinations.json"
log="$root/${variant}_cmu_eval.log"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled

exec > >(tee -a "$log") 2>&1
echo "START $(date --iso-8601=seconds)"
while [[ ! -f "$root/$variant/final_state.pth.tar" ]]; do
  sleep 15
done
"$python" "$repo/RUMPL/run/eval_exact_temporal_cmu.py" \
  --temporal-checkpoint "$checkpoint" \
  --annotation "$annotation" \
  --output "$output" \
  --batch-size 32 --workers 4
echo "END $(date --iso-8601=seconds)"
