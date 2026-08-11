#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-1}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/temporal_exact_20260723
variant=T3_r5_motiondiff_l9_oks05_grouped_allviews_rp005_seed0_20260723
config="$repo/RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
checkpoint=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar
data='/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl'
python=/home/lixiaob/cjy/rumpl_venv310/bin/python

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled

mkdir -p "$root/$variant"
exec > >(tee -a "$root/${variant}.log") 2>&1
echo "START $(date --iso-8601=seconds)"
cp "$repo/RUMPL/lib/models/exact_temporal_rumpl.py" "$root/$variant/"
cp "$repo/RUMPL/run/train_exact_temporal_rumpl.py" "$root/$variant/"
"$python" "$repo/RUMPL/run/train_exact_temporal_rumpl.py" \
  --config "$config" --checkpoint "$checkpoint" --data "$data" \
  --output "$root/$variant" \
  --frames 9 --depth 2 --epochs 20 --batch-size 16 --workers 4 \
  --validation-clips 200 --learning-rate 4e-5 --weight-decay 0.1 \
  --huber-delta 0.1 --min-center-oks 0.5 --seed 0 \
  --motion-only --residual-penalty 0.05
echo "END $(date --iso-8601=seconds)"
